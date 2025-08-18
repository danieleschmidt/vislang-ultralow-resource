#!/usr/bin/env python3
"""
Automated dependency update checker for VisLang-UltraLow-Resource project.

This script checks for available dependency updates and creates structured 
reports for security and maintenance updates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import tempfile
import re

try:
    import requests
    import toml
    from packaging.version import Version, parse
    from packaging.specifiers import SpecifierSet
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install requests toml packaging")
    sys.exit(1)


class DependencyUpdater:
    """Automated dependency update checker and manager."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.requirements_files = [
            self.project_root / "requirements.txt",
            self.project_root / "requirements-dev.txt",
            self.project_root / "requirements-test.txt"
        ]
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # GitHub token for API access
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "danieleschmidt")
        self.repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "vislang-ultralow-resource")

    def load_pyproject_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Load dependencies from pyproject.toml."""
        if not self.pyproject_file.exists():
            self.logger.warning("pyproject.toml not found")
            return {}

        try:
            with open(self.pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)

            dependencies = {}
            
            # Main dependencies
            if 'project' in pyproject_data and 'dependencies' in pyproject_data['project']:
                dependencies['main'] = self._parse_dependencies(
                    pyproject_data['project']['dependencies']
                )

            # Optional dependencies
            if 'project' in pyproject_data and 'optional-dependencies' in pyproject_data['project']:
                for group, deps in pyproject_data['project']['optional-dependencies'].items():
                    dependencies[f'optional-{group}'] = self._parse_dependencies(deps)

            # Build system dependencies
            if 'build-system' in pyproject_data and 'requires' in pyproject_data['build-system']:
                dependencies['build'] = self._parse_dependencies(
                    pyproject_data['build-system']['requires']
                )

            return dependencies

        except Exception as e:
            self.logger.error(f"Error loading pyproject.toml: {e}")
            return {}

    def load_requirements_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Load dependencies from requirements files."""
        dependencies = {}
        
        for req_file in self.requirements_files:
            if not req_file.exists():
                continue
                
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                
                file_deps = {}
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        parsed = self._parse_requirement_line(line)
                        if parsed:
                            file_deps[parsed[0]] = parsed[1]
                
                dependencies[req_file.name] = file_deps
                
            except Exception as e:
                self.logger.error(f"Error loading {req_file}: {e}")
                
        return dependencies

    def _parse_dependencies(self, deps: List[str]) -> Dict[str, str]:
        """Parse dependency list into name-version mapping."""
        parsed = {}
        for dep in deps:
            parsed_req = self._parse_requirement_line(dep)
            if parsed_req:
                parsed[parsed_req[0]] = parsed_req[1]
        return parsed

    def _parse_requirement_line(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse a single requirement line."""
        # Remove comments
        line = line.split('#')[0].strip()
        if not line:
            return None
            
        # Handle different requirement formats
        patterns = [
            r'^([a-zA-Z0-9_-]+)([><=!]+[0-9.]+.*)?$',  # name>=1.0.0
            r'^([a-zA-Z0-9_-]+)\[.*\]([><=!]+[0-9.]+.*)?$',  # name[extra]>=1.0.0
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                version_spec = match.group(2) if match.group(2) else ""
                return (name, version_spec)
                
        return None

    def get_latest_versions(self, package_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get latest versions for packages from PyPI."""
        latest_versions = {}
        
        for package_name in package_names:
            try:
                url = f"https://pypi.org/pypi/{package_name}/json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    info = data.get('info', {})
                    
                    latest_versions[package_name] = {
                        'latest_version': info.get('version', ''),
                        'summary': info.get('summary', ''),
                        'author': info.get('author', ''),
                        'license': info.get('license', ''),
                        'home_page': info.get('home_page', ''),
                        'release_date': self._get_release_date(data, info.get('version', ''))
                    }
                    
            except Exception as e:
                self.logger.warning(f"Failed to get info for {package_name}: {e}")
                latest_versions[package_name] = {
                    'latest_version': 'unknown',
                    'error': str(e)
                }
                
        return latest_versions

    def _get_release_date(self, pypi_data: Dict, version: str) -> str:
        """Get release date for a specific version."""
        try:
            releases = pypi_data.get('releases', {})
            if version in releases and releases[version]:
                return releases[version][0].get('upload_time_iso_8601', '')
        except Exception:
            pass
        return ''

    def check_security_advisories(self, package_names: List[str]) -> Dict[str, List[Dict]]:
        """Check for security advisories using GitHub's advisory database."""
        advisories = {}
        
        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not provided, skipping security advisory check")
            return advisories
            
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        for package_name in package_names:
            try:
                # GitHub Security Advisory API
                url = f"https://api.github.com/advisories"
                params = {
                    "ecosystem": "pip",
                    "affects": package_name,
                    "per_page": 10
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    advisories[package_name] = [
                        {
                            'id': advisory.get('ghsa_id', ''),
                            'summary': advisory.get('summary', ''),
                            'severity': advisory.get('severity', ''),
                            'published_at': advisory.get('published_at', ''),
                            'updated_at': advisory.get('updated_at', ''),
                            'withdrawn_at': advisory.get('withdrawn_at'),
                            'vulnerabilities': advisory.get('vulnerabilities', [])
                        }
                        for advisory in data
                    ]
                    
            except Exception as e:
                self.logger.warning(f"Failed to check advisories for {package_name}: {e}")
                
        return advisories

    def analyze_updates(self, dependencies: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """Analyze available updates for dependencies."""
        all_packages = set()
        for dep_group in dependencies.values():
            all_packages.update(dep_group.keys())
        
        all_packages = list(all_packages)
        self.logger.info(f"Checking updates for {len(all_packages)} packages...")
        
        # Get latest versions
        latest_versions = self.get_latest_versions(all_packages)
        
        # Check security advisories
        advisories = self.check_security_advisories(all_packages)
        
        # Analyze updates
        analysis = {
            'packages_checked': len(all_packages),
            'updates_available': [],
            'security_updates': [],
            'major_updates': [],
            'minor_updates': [],
            'patch_updates': [],
            'errors': []
        }
        
        for package_name in all_packages:
            if package_name not in latest_versions:
                continue
                
            package_info = latest_versions[package_name]
            
            if 'error' in package_info:
                analysis['errors'].append({
                    'package': package_name,
                    'error': package_info['error']
                })
                continue
                
            latest_version = package_info['latest_version']
            
            # Find current version from dependencies
            current_version = None
            dep_group = None
            
            for group_name, group_deps in dependencies.items():
                if package_name in group_deps:
                    current_version_spec = group_deps[package_name]
                    current_version = self._extract_version_from_spec(current_version_spec)
                    dep_group = group_name
                    break
            
            if not current_version or not latest_version:
                continue
                
            try:
                current_ver = parse(current_version)
                latest_ver = parse(latest_version)
                
                if latest_ver > current_ver:
                    update_info = {
                        'package': package_name,
                        'current_version': current_version,
                        'latest_version': latest_version,
                        'dependency_group': dep_group,
                        'release_date': package_info.get('release_date', ''),
                        'summary': package_info.get('summary', ''),
                        'has_security_advisory': package_name in advisories and bool(advisories[package_name])
                    }
                    
                    # Categorize update type
                    if current_ver.major != latest_ver.major:
                        update_info['update_type'] = 'major'
                        analysis['major_updates'].append(update_info)
                    elif current_ver.minor != latest_ver.minor:
                        update_info['update_type'] = 'minor'
                        analysis['minor_updates'].append(update_info)
                    else:
                        update_info['update_type'] = 'patch'
                        analysis['patch_updates'].append(update_info)
                    
                    # Add security information
                    if package_name in advisories and advisories[package_name]:
                        update_info['security_advisories'] = advisories[package_name]
                        analysis['security_updates'].append(update_info)
                    
                    analysis['updates_available'].append(update_info)
                    
            except Exception as e:
                self.logger.warning(f"Failed to compare versions for {package_name}: {e}")
                
        return analysis

    def _extract_version_from_spec(self, version_spec: str) -> Optional[str]:
        """Extract version number from version specifier."""
        if not version_spec:
            return None
            
        # Remove operators and extract version
        version_pattern = r'([0-9]+\.[0-9]+[0-9.]*)'
        match = re.search(version_pattern, version_spec)
        
        if match:
            return match.group(1)
            
        return None

    def generate_update_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a formatted update report."""
        report = f"""# Dependency Update Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Packages Checked**: {analysis['packages_checked']}
**Updates Available**: {len(analysis['updates_available'])}

## Summary

- 🔴 **Security Updates**: {len(analysis['security_updates'])}
- 🟡 **Major Updates**: {len(analysis['major_updates'])}
- 🟢 **Minor Updates**: {len(analysis['minor_updates'])}
- 🔵 **Patch Updates**: {len(analysis['patch_updates'])}

"""

        if analysis['security_updates']:
            report += "## 🚨 Security Updates (High Priority)\n\n"
            for update in analysis['security_updates']:
                report += f"- **{update['package']}**: {update['current_version']} → {update['latest_version']}\n"
                if 'security_advisories' in update:
                    for advisory in update['security_advisories']:
                        report += f"  - ⚠️ {advisory['severity'].upper()}: {advisory['summary']}\n"
                report += "\n"

        if analysis['major_updates']:
            report += "## 🔴 Major Updates (Review Required)\n\n"
            for update in analysis['major_updates']:
                if update not in analysis['security_updates']:
                    report += f"- **{update['package']}**: {update['current_version']} → {update['latest_version']}\n"
                    report += f"  - Group: {update['dependency_group']}\n"
                    if update['summary']:
                        report += f"  - Summary: {update['summary']}\n"
                    report += "\n"

        if analysis['minor_updates']:
            report += "## 🟢 Minor Updates (Generally Safe)\n\n"
            for update in analysis['minor_updates']:
                if update not in analysis['security_updates']:
                    report += f"- **{update['package']}**: {update['current_version']} → {update['latest_version']}\n"

        if analysis['patch_updates']:
            report += "\n## 🔵 Patch Updates (Safe to Apply)\n\n"
            for update in analysis['patch_updates']:
                if update not in analysis['security_updates']:
                    report += f"- **{update['package']}**: {update['current_version']} → {update['latest_version']}\n"

        if analysis['errors']:
            report += "\n## ❌ Errors\n\n"
            for error in analysis['errors']:
                report += f"- **{error['package']}**: {error['error']}\n"

        report += f"\n---\n*Report generated by dependency-updater.py*\n"

        return report

    def create_github_issue(self, analysis: Dict[str, Any]) -> bool:
        """Create a GitHub issue with the dependency update report."""
        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not provided, cannot create GitHub issue")
            return False

        # Only create issue if there are security updates or many updates
        if not analysis['security_updates'] and len(analysis['updates_available']) < 5:
            self.logger.info("No significant updates found, skipping issue creation")
            return False

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        title = f"Dependency Updates Available - {datetime.now().strftime('%Y-%m-%d')}"
        
        if analysis['security_updates']:
            title = f"🚨 Security Updates Required - {datetime.now().strftime('%Y-%m-%d')}"

        body = self.generate_update_report(analysis)
        body += "\n\n## Recommended Actions\n\n"
        
        if analysis['security_updates']:
            body += "- [ ] **URGENT**: Review and apply security updates immediately\n"
        
        if analysis['patch_updates']:
            body += "- [ ] Apply patch updates (low risk)\n"
            
        if analysis['minor_updates']:
            body += "- [ ] Review and test minor updates\n"
            
        if analysis['major_updates']:
            body += "- [ ] Carefully review major updates (may contain breaking changes)\n"
            
        body += "- [ ] Run full test suite after updates\n"
        body += "- [ ] Update lock files and documentation\n"

        issue_data = {
            "title": title,
            "body": body,
            "labels": ["dependencies", "maintenance"] + (["security"] if analysis['security_updates'] else [])
        }

        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues"
            response = requests.post(url, headers=headers, json=issue_data, timeout=10)
            
            if response.status_code == 201:
                issue_url = response.json().get('html_url', '')
                self.logger.info(f"Created GitHub issue: {issue_url}")
                return True
            else:
                self.logger.error(f"Failed to create issue: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error creating GitHub issue: {e}")
            
        return False

    def save_report(self, analysis: Dict[str, Any], output_file: str) -> bool:
        """Save the analysis report to a file."""
        try:
            report = self.generate_update_report(analysis)
            
            with open(output_file, 'w') as f:
                f.write(report)
                
            self.logger.info(f"Report saved to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return False

    def run_dependency_check(self, create_issue: bool = False, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete dependency check process."""
        self.logger.info("Starting dependency update check...")
        
        # Load dependencies from all sources
        pyproject_deps = self.load_pyproject_dependencies()
        requirements_deps = self.load_requirements_dependencies()
        
        all_dependencies = {**pyproject_deps, **requirements_deps}
        
        if not all_dependencies:
            self.logger.warning("No dependencies found to check")
            return {}
            
        # Analyze updates
        analysis = self.analyze_updates(all_dependencies)
        
        # Generate and save report
        if output_file:
            self.save_report(analysis, output_file)
            
        # Create GitHub issue if requested
        if create_issue:
            self.create_github_issue(analysis)
            
        # Print summary
        self.logger.info(f"Dependency check completed:")
        self.logger.info(f"  - Packages checked: {analysis['packages_checked']}")
        self.logger.info(f"  - Updates available: {len(analysis['updates_available'])}")
        self.logger.info(f"  - Security updates: {len(analysis['security_updates'])}")
        self.logger.info(f"  - Major updates: {len(analysis['major_updates'])}")
        
        return analysis


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Check for dependency updates")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project"
    )
    parser.add_argument(
        "--output",
        help="Output file for the update report"
    )
    parser.add_argument(
        "--create-issue",
        action="store_true",
        help="Create a GitHub issue with the update report"
    )
    parser.add_argument(
        "--json",
        help="Save analysis as JSON to specified file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize updater
    updater = DependencyUpdater(args.project_root)

    # Run dependency check
    analysis = updater.run_dependency_check(
        create_issue=args.create_issue,
        output_file=args.output
    )

    # Save JSON analysis if requested
    if args.json:
        try:
            with open(args.json, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            print(f"Analysis saved to {args.json}")
        except Exception as e:
            print(f"Failed to save JSON analysis: {e}")

    # Exit with appropriate code
    if analysis.get('security_updates'):
        sys.exit(2)  # Security updates required
    elif analysis.get('updates_available'):
        sys.exit(1)  # Updates available
    else:
        sys.exit(0)  # No updates


if __name__ == "__main__":
    main()