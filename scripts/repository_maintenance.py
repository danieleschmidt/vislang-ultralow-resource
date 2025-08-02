#!/usr/bin/env python3
"""
Repository maintenance script for VisLang-UltraLow-Resource project.

This script performs various maintenance tasks like cleaning up old branches,
updating documentation, checking for issues, and generating reports.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
import re

try:
    import requests
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install requests")
    sys.exit(1)


class RepositoryMaintainer:
    """Handles repository maintenance tasks."""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "danieleschmidt")
        self.repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "vislang-ultralow-resource")
        self.repo_path = Path.cwd()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def cleanup_old_branches(self, days_old: int = 30, dry_run: bool = True) -> List[str]:
        """Clean up old merged branches."""
        cleaned_branches = []
        
        try:
            # Get merged branches
            result = subprocess.run(
                ["git", "branch", "--merged", "main"],
                capture_output=True,
                text=True,
                check=True
            )
            
            merged_branches = [
                branch.strip().replace('* ', '') 
                for branch in result.stdout.split('\n') 
                if branch.strip() and not branch.strip().endswith('main')
            ]

            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            for branch in merged_branches:
                if branch in ['main', 'master', 'develop']:
                    continue
                    
                try:
                    # Get last commit date for branch
                    result = subprocess.run(
                        ["git", "log", "-1", "--format=%ci", branch],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    last_commit_date = datetime.strptime(
                        result.stdout.strip().split()[0], '%Y-%m-%d'
                    )
                    
                    if last_commit_date < cutoff_date:
                        if not dry_run:
                            subprocess.run(
                                ["git", "branch", "-d", branch],
                                check=True
                            )
                            self.logger.info(f"Deleted old branch: {branch}")
                        else:
                            self.logger.info(f"Would delete old branch: {branch}")
                        
                        cleaned_branches.append(branch)
                        
                except subprocess.CalledProcessError:
                    self.logger.warning(f"Could not process branch: {branch}")

            self.logger.info(f"{'Would clean' if dry_run else 'Cleaned'} {len(cleaned_branches)} old branches")
            return cleaned_branches

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error cleaning branches: {e}")
            return []

    def update_readme_badges(self) -> bool:
        """Update badges in README.md with current information."""
        readme_path = self.repo_path / "README.md"
        
        if not readme_path.exists():
            self.logger.warning("README.md not found")
            return False

        try:
            with open(readme_path, 'r') as f:
                content = f.read()

            # Update various badges
            updates_made = False

            # Python version badge
            if 'python-' in content:
                # Extract Python versions from pyproject.toml or requirements
                python_versions = self._get_python_versions()
                if python_versions:
                    new_python_badge = f"![Python {python_versions}](https://img.shields.io/badge/python-{python_versions}-blue.svg)"
                    content = re.sub(
                        r'!\[Python [^\]]+\]\([^)]+\)',
                        new_python_badge,
                        content
                    )
                    updates_made = True

            # License badge
            license_file = self.repo_path / "LICENSE"
            if license_file.exists():
                license_type = self._detect_license_type()
                if license_type:
                    new_license_badge = f"![License {license_type}](https://img.shields.io/badge/license-{license_type}-green.svg)"
                    content = re.sub(
                        r'!\[License [^\]]+\]\([^)]+\)',
                        new_license_badge,
                        content
                    )
                    updates_made = True

            # Tests status badge (if using GitHub Actions)
            if self.github_token:
                new_tests_badge = f"![Tests](https://github.com/{self.repo_owner}/{self.repo_name}/workflows/CI/badge.svg)"
                content = re.sub(
                    r'!\[Tests[^\]]*\]\([^)]+\)',
                    new_tests_badge,
                    content
                )
                updates_made = True

            if updates_made:
                with open(readme_path, 'w') as f:
                    f.write(content)
                self.logger.info("Updated README.md badges")
                return True
            else:
                self.logger.info("No badge updates needed")
                return False

        except Exception as e:
            self.logger.error(f"Error updating README badges: {e}")
            return False

    def _get_python_versions(self) -> Optional[str]:
        """Extract Python versions from project configuration."""
        try:
            # Check pyproject.toml
            pyproject_path = self.repo_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    
                # Look for requires-python
                match = re.search(r'requires-python\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1).replace('>=', '').replace('~=', '')

            # Fallback to checking CI configuration
            ci_path = self.repo_path / ".github" / "workflows" / "ci.yml"
            if ci_path.exists():
                with open(ci_path, 'r') as f:
                    content = f.read()
                    
                # Look for Python versions in matrix
                versions = re.findall(r'[\'"](3\.\d+)[\'"]', content)
                if versions:
                    return f"{min(versions)}+"

            return "3.8+"

        except Exception:
            return None

    def _detect_license_type(self) -> Optional[str]:
        """Detect license type from LICENSE file."""
        try:
            license_path = self.repo_path / "LICENSE"
            if license_path.exists():
                with open(license_path, 'r') as f:
                    content = f.read().lower()

                if 'mit license' in content:
                    return 'MIT'
                elif 'apache license' in content:
                    return 'Apache--2.0'
                elif 'gnu general public license' in content:
                    return 'GPL--3.0'
                elif 'bsd' in content:
                    return 'BSD--3--Clause'

            return 'Unknown'

        except Exception:
            return None

    def check_broken_links(self) -> List[str]:
        """Check for broken links in documentation."""
        broken_links = []
        
        # Find all markdown files
        md_files = list(self.repo_path.rglob("*.md"))
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find all links
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                
                for link_text, link_url in links:
                    if link_url.startswith('http'):
                        # Check external links
                        try:
                            response = requests.head(link_url, timeout=10, allow_redirects=True)
                            if response.status_code >= 400:
                                broken_links.append(f"{md_file}: {link_url} (HTTP {response.status_code})")
                        except requests.RequestException:
                            broken_links.append(f"{md_file}: {link_url} (Connection error)")
                    
                    elif not link_url.startswith('#'):
                        # Check internal links
                        if link_url.startswith('/'):
                            target_path = self.repo_path / link_url.lstrip('/')
                        else:
                            target_path = md_file.parent / link_url

                        if not target_path.exists():
                            broken_links.append(f"{md_file}: {link_url} (File not found)")

            except Exception as e:
                self.logger.warning(f"Error checking links in {md_file}: {e}")

        if broken_links:
            self.logger.warning(f"Found {len(broken_links)} broken links")
        else:
            self.logger.info("No broken links found")

        return broken_links

    def validate_project_structure(self) -> List[str]:
        """Validate that the project has the expected structure."""
        issues = []
        
        # Required files
        required_files = [
            "README.md",
            "LICENSE",
            "pyproject.toml",
            "requirements.txt",
            ".gitignore",
            "CONTRIBUTING.md",
            "SECURITY.md"
        ]

        for file_name in required_files:
            if not (self.repo_path / file_name).exists():
                issues.append(f"Missing required file: {file_name}")

        # Required directories
        required_dirs = [
            "src",
            "tests",
            "docs",
            ".github"
        ]

        for dir_name in required_dirs:
            if not (self.repo_path / dir_name).exists():
                issues.append(f"Missing required directory: {dir_name}")

        # Check for Python package structure
        src_dir = self.repo_path / "src"
        if src_dir.exists():
            python_packages = list(src_dir.glob("*/"))
            if not python_packages:
                issues.append("No Python packages found in src/")
            else:
                for package_dir in python_packages:
                    if not (package_dir / "__init__.py").exists():
                        issues.append(f"Missing __init__.py in {package_dir}")

        # Check test structure
        tests_dir = self.repo_path / "tests"
        if tests_dir.exists():
            test_categories = ["unit", "integration", "e2e"]
            for category in test_categories:
                if not (tests_dir / category).exists():
                    issues.append(f"Missing test directory: tests/{category}")

        if issues:
            self.logger.warning(f"Found {len(issues)} project structure issues")
        else:
            self.logger.info("Project structure validation passed")

        return issues

    def generate_maintenance_report(self) -> str:
        """Generate a comprehensive maintenance report."""
        self.logger.info("Generating maintenance report...")

        # Collect data
        old_branches = self.cleanup_old_branches(dry_run=True)
        broken_links = self.check_broken_links()
        structure_issues = self.validate_project_structure()

        # Git statistics
        try:
            # Get commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            total_commits = int(result.stdout.strip())

            # Get contributor count
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--all"],
                capture_output=True,
                text=True,
                check=True
            )
            contributors = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

            # Get recent activity
            result = subprocess.run(
                ["git", "log", "--since=30 days ago", "--oneline"],
                capture_output=True,
                text=True,
                check=True
            )
            recent_commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0

        except subprocess.CalledProcessError:
            total_commits = contributors = recent_commits = 0

        # Generate report
        report = f"""# Repository Maintenance Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Repository**: {self.repo_owner}/{self.repo_name}

## Overview

- 📊 Total Commits: {total_commits}
- 👥 Contributors: {contributors}
- 🔄 Recent Activity (30 days): {recent_commits} commits

## Maintenance Issues

### Branch Cleanup
- 🌿 Old Merged Branches: {len(old_branches)}
{chr(10).join(f"  - {branch}" for branch in old_branches[:5])}
{'  - ...' if len(old_branches) > 5 else ''}

### Documentation
- 🔗 Broken Links: {len(broken_links)}
{chr(10).join(f"  - {link}" for link in broken_links[:3])}
{'  - ...' if len(broken_links) > 3 else ''}

### Project Structure
- ⚠️ Structure Issues: {len(structure_issues)}
{chr(10).join(f"  - {issue}" for issue in structure_issues[:5])}
{'  - ...' if len(structure_issues) > 5 else ''}

## Recommendations

### Immediate Actions
{self._generate_recommendations(old_branches, broken_links, structure_issues)}

### Automation Opportunities
- Set up automated branch cleanup
- Implement link checking in CI/CD
- Add project structure validation
- Configure automated badge updates

## Health Score

{self._calculate_health_score(old_branches, broken_links, structure_issues)}

---
*This report was generated automatically by the repository maintenance script.*
"""

        return report

    def _generate_recommendations(self, old_branches: List[str], broken_links: List[str], structure_issues: List[str]) -> str:
        """Generate maintenance recommendations."""
        recommendations = []

        if old_branches:
            recommendations.append(f"1. Clean up {len(old_branches)} old merged branches")

        if broken_links:
            recommendations.append(f"2. Fix {len(broken_links)} broken links in documentation")

        if structure_issues:
            recommendations.append(f"3. Address {len(structure_issues)} project structure issues")

        if not recommendations:
            recommendations.append("1. Repository is well-maintained, no immediate actions needed")

        recommendations.extend([
            f"{len(recommendations) + 1}. Update README badges with current information",
            f"{len(recommendations) + 2}. Run security audit on dependencies",
            f"{len(recommendations) + 3}. Review and update documentation"
        ])

        return '\n'.join(recommendations)

    def _calculate_health_score(self, old_branches: List[str], broken_links: List[str], structure_issues: List[str]) -> str:
        """Calculate repository health score."""
        # Start with perfect score
        score = 100

        # Deduct points for issues
        score -= len(old_branches) * 2  # 2 points per old branch
        score -= len(broken_links) * 5  # 5 points per broken link
        score -= len(structure_issues) * 10  # 10 points per structure issue

        # Ensure score doesn't go below 0
        score = max(0, score)

        if score >= 90:
            status = "🟢 Excellent"
        elif score >= 70:
            status = "🟡 Good"
        elif score >= 50:
            status = "🟠 Needs Attention"
        else:
            status = "🔴 Poor"

        return f"**Overall Health**: {score}/100 {status}"

    def run_maintenance(self, tasks: List[str], dry_run: bool = True) -> Dict[str, Any]:
        """Run specified maintenance tasks."""
        results = {}

        if "cleanup-branches" in tasks:
            results["cleanup_branches"] = self.cleanup_old_branches(dry_run=dry_run)

        if "update-badges" in tasks:
            results["update_badges"] = self.update_readme_badges()

        if "check-links" in tasks:
            results["broken_links"] = self.check_broken_links()

        if "validate-structure" in tasks:
            results["structure_issues"] = self.validate_project_structure()

        if "generate-report" in tasks:
            results["report"] = self.generate_maintenance_report()

        return results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Repository maintenance tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["cleanup-branches", "update-badges", "check-links", "validate-structure", "generate-report"],
        default=["generate-report"],
        help="Maintenance tasks to run"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--output-report",
        help="Output report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize maintainer
    maintainer = RepositoryMaintainer()

    # Run maintenance tasks
    results = maintainer.run_maintenance(args.tasks, dry_run=args.dry_run)

    # Output results
    if "report" in results:
        if args.output_report:
            with open(args.output_report, 'w') as f:
                f.write(results["report"])
            print(f"Maintenance report saved to {args.output_report}")
        else:
            print(results["report"])

    # Summary
    print(f"\nMaintenance completed. Tasks run: {', '.join(args.tasks)}")


if __name__ == "__main__":
    main()