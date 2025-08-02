#!/usr/bin/env python3
"""
Automated dependency update script for VisLang-UltraLow-Resource project.

This script checks for dependency updates, categorizes them by severity,
and can automatically create pull requests for updates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import logging
import re

try:
    import requests
    from packaging import version
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install requests packaging")
    sys.exit(1)


class DependencyUpdater:
    """Handles dependency updates and security checks."""
    
    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = Path(requirements_file)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "danieleschmidt")
        self.repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "vislang-ultralow-resource")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_current_dependencies(self) -> Dict[str, str]:
        """Parse current dependencies from requirements file."""
        dependencies = {}
        
        if not self.requirements_file.exists():
            self.logger.error(f"Requirements file not found: {self.requirements_file}")
            return dependencies

        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        # Parse package name and version
                        match = re.match(r'^([a-zA-Z0-9_-]+)([>=<~!]+)([0-9.]+)', line)
                        if match:
                            package_name = match.group(1)
                            operator = match.group(2)
                            package_version = match.group(3)
                            dependencies[package_name] = package_version
                        elif '==' in line:
                            package_name, package_version = line.split('==')
                            dependencies[package_name.strip()] = package_version.strip()

            self.logger.info(f"Found {len(dependencies)} dependencies")
            return dependencies

        except Exception as e:
            self.logger.error(f"Error parsing requirements file: {e}")
            return {}

    def check_outdated_packages(self) -> List[Dict[str, Any]]:
        """Check for outdated packages using pip list --outdated."""
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            outdated_packages = json.loads(result.stdout)
            self.logger.info(f"Found {len(outdated_packages)} outdated packages")
            return outdated_packages

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking outdated packages: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing pip output: {e}")
            return []

    def check_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities using pip-audit."""
        vulnerabilities = []
        
        try:
            # Try pip-audit first
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get("vulnerabilities", [])
                self.logger.info(f"Found {len(vulnerabilities)} vulnerabilities via pip-audit")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("pip-audit not available, trying safety")
            
            try:
                # Fallback to safety
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data.get("vulnerabilities", [])
                    self.logger.info(f"Found {len(vulnerabilities)} vulnerabilities via safety")
                    
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
                self.logger.warning("Neither pip-audit nor safety available for security checking")

        return vulnerabilities

    def categorize_updates(self, outdated_packages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize updates by severity (patch, minor, major)."""
        categories = {
            "patch": [],
            "minor": [],
            "major": []
        }

        for package in outdated_packages:
            try:
                current_version = version.parse(package["version"])
                latest_version = version.parse(package["latest_version"])

                if latest_version.major > current_version.major:
                    categories["major"].append(package)
                elif latest_version.minor > current_version.minor:
                    categories["minor"].append(package)
                else:
                    categories["patch"].append(package)

            except Exception as e:
                self.logger.warning(f"Error parsing version for {package['name']}: {e}")
                # Default to minor if we can't parse
                categories["minor"].append(package)

        for category, packages in categories.items():
            self.logger.info(f"{category.title()} updates: {len(packages)}")

        return categories

    def update_requirements_file(self, packages_to_update: List[str]) -> bool:
        """Update requirements file with new package versions."""
        if not packages_to_update:
            self.logger.info("No packages to update")
            return False

        try:
            # Create backup
            backup_file = self.requirements_file.with_suffix('.txt.backup')
            with open(self.requirements_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())

            # Get latest versions for packages to update
            updated_versions = {}
            for package in packages_to_update:
                try:
                    result = subprocess.run(
                        ["pip", "index", "versions", package],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        # Parse output to get latest version
                        lines = result.stdout.split('\n')
                        for line in lines:
                            if "Available versions:" in line:
                                versions = line.split(":")[1].strip().split(", ")
                                if versions and versions[0]:
                                    updated_versions[package] = versions[0]
                                break
                except Exception as e:
                    self.logger.warning(f"Could not get latest version for {package}: {e}")

            # Update requirements file
            updated_lines = []
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        for package in packages_to_update:
                            if line.startswith(package):
                                if package in updated_versions:
                                    line = f"{package}>={updated_versions[package]}"
                                break
                    updated_lines.append(line)

            with open(self.requirements_file, 'w') as f:
                f.write('\n'.join(updated_lines))

            self.logger.info(f"Updated {len(packages_to_update)} packages in {self.requirements_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating requirements file: {e}")
            # Restore backup if it exists
            if backup_file.exists():
                with open(backup_file, 'r') as src, open(self.requirements_file, 'w') as dst:
                    dst.write(src.read())
                self.logger.info("Restored requirements file from backup")
            return False

    def create_update_branch(self, branch_name: str, packages: List[str]) -> bool:
        """Create a new git branch for updates."""
        try:
            # Check if we're in a git repository
            subprocess.run(["git", "status"], check=True, capture_output=True)

            # Create and checkout new branch
            subprocess.run(["git", "checkout", "-b", branch_name], check=True)
            
            # Update requirements file
            if self.update_requirements_file(packages):
                # Add and commit changes
                subprocess.run(["git", "add", str(self.requirements_file)], check=True)
                commit_message = f"deps: update {len(packages)} packages\n\nUpdated packages:\n" + \
                               "\n".join(f"- {pkg}" for pkg in packages)
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                
                self.logger.info(f"Created branch {branch_name} with updates")
                return True
            else:
                # Cleanup branch if update failed
                subprocess.run(["git", "checkout", "-"], check=True)
                subprocess.run(["git", "branch", "-D", branch_name], check=True)
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error creating update branch: {e}")
            return False

    def create_pull_request(self, branch_name: str, title: str, body: str) -> bool:
        """Create a pull request for the update branch."""
        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not provided, cannot create pull request")
            return False

        try:
            # Push branch to remote
            subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)

            # Create pull request via GitHub API
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "title": title,
                "body": body,
                "head": branch_name,
                "base": "main"
            }

            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 201:
                pr_data = response.json()
                self.logger.info(f"Created pull request: {pr_data['html_url']}")
                return True
            else:
                self.logger.error(f"Failed to create pull request: {response.text}")
                return False

        except Exception as e:
            self.logger.error(f"Error creating pull request: {e}")
            return False

    def run_security_updates(self, auto_merge: bool = False) -> bool:
        """Run security updates automatically."""
        vulnerabilities = self.check_security_vulnerabilities()
        
        if not vulnerabilities:
            self.logger.info("No security vulnerabilities found")
            return True

        # Extract package names from vulnerabilities
        vulnerable_packages = []
        for vuln in vulnerabilities:
            package_name = vuln.get("package", vuln.get("name", ""))
            if package_name and package_name not in vulnerable_packages:
                vulnerable_packages.append(package_name)

        if not vulnerable_packages:
            self.logger.info("No vulnerable packages to update")
            return True

        self.logger.info(f"Found {len(vulnerable_packages)} vulnerable packages: {vulnerable_packages}")

        # Create security update branch
        branch_name = f"security/dependency-updates-{datetime.now().strftime('%Y%m%d')}"
        
        if self.create_update_branch(branch_name, vulnerable_packages):
            # Create pull request
            title = "🔒 Security: Update vulnerable dependencies"
            body = f"""## Security Dependency Updates

This PR updates dependencies with known security vulnerabilities.

### Vulnerabilities Fixed
{chr(10).join(f"- {vuln.get('package', vuln.get('name', 'Unknown'))}: {vuln.get('id', 'N/A')}" for vuln in vulnerabilities)}

### Updated Packages
{chr(10).join(f"- {pkg}" for pkg in vulnerable_packages)}

**Auto-merge**: {'This PR will be automatically merged after CI passes.' if auto_merge else 'Manual review required.'}

Generated by automated dependency update script.
"""
            
            if self.create_pull_request(branch_name, title, body):
                self.logger.info("Security update pull request created successfully")
                return True

        return False

    def run_regular_updates(self, update_type: str = "minor") -> bool:
        """Run regular dependency updates."""
        outdated_packages = self.check_outdated_packages()
        
        if not outdated_packages:
            self.logger.info("All packages are up to date")
            return True

        categories = self.categorize_updates(outdated_packages)
        
        packages_to_update = []
        if update_type == "patch":
            packages_to_update = [pkg["name"] for pkg in categories["patch"]]
        elif update_type == "minor":
            packages_to_update = [pkg["name"] for pkg in categories["patch"] + categories["minor"]]
        elif update_type == "major":
            packages_to_update = [pkg["name"] for pkg in outdated_packages]

        if not packages_to_update:
            self.logger.info(f"No {update_type} updates available")
            return True

        # Create update branch
        branch_name = f"deps/{update_type}-updates-{datetime.now().strftime('%Y%m%d')}"
        
        if self.create_update_branch(branch_name, packages_to_update):
            # Create pull request
            title = f"⬆️ Update dependencies ({update_type} updates)"
            body = f"""## {update_type.title()} Dependency Updates

This PR updates dependencies to their latest {update_type} versions.

### Updated Packages ({len(packages_to_update)})
{chr(10).join(f"- {pkg}" for pkg in packages_to_update)}

### Review Required
{'Please review these changes before merging.' if update_type != 'patch' else 'These are patch updates and should be safe to merge.'}

Generated by automated dependency update script.
"""
            
            if self.create_pull_request(branch_name, title, body):
                self.logger.info(f"{update_type.title()} update pull request created successfully")
                return True

        return False

    def generate_update_report(self) -> str:
        """Generate a summary report of available updates."""
        outdated_packages = self.check_outdated_packages()
        vulnerabilities = self.check_security_vulnerabilities()
        categories = self.categorize_updates(outdated_packages)

        report = f"""
# Dependency Update Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Security Status

- 🔒 Vulnerabilities Found: {len(vulnerabilities)}
- ⚠️ Vulnerable Packages: {len(set(v.get('package', v.get('name', '')) for v in vulnerabilities))}

## Available Updates

- 🔧 Patch Updates: {len(categories['patch'])}
- ⬆️ Minor Updates: {len(categories['minor'])}
- 🚀 Major Updates: {len(categories['major'])}
- 📦 Total Outdated: {len(outdated_packages)}

## Update Recommendations

1. **Immediate Action Required** (Security):
{chr(10).join(f"   - {v.get('package', v.get('name', 'Unknown'))}" for v in vulnerabilities[:5])}

2. **Safe to Update** (Patch):
{chr(10).join(f"   - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}" for pkg in categories['patch'][:5])}

3. **Review Required** (Minor):
{chr(10).join(f"   - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}" for pkg in categories['minor'][:5])}

4. **Manual Review Required** (Major):
{chr(10).join(f"   - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}" for pkg in categories['major'][:5])}
"""
        return report


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Automated dependency updates")
    parser.add_argument(
        "--action",
        choices=["check", "security", "update", "report"],
        default="check",
        help="Action to perform"
    )
    parser.add_argument(
        "--update-type",
        choices=["patch", "minor", "major"],
        default="minor",
        help="Type of updates to apply"
    )
    parser.add_argument(
        "--requirements-file",
        default="requirements.txt",
        help="Path to requirements file"
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help="Enable auto-merge for security updates"
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

    # Initialize updater
    updater = DependencyUpdater(args.requirements_file)

    # Perform requested action
    if args.action == "check":
        outdated = updater.check_outdated_packages()
        vulnerabilities = updater.check_security_vulnerabilities()
        print(f"Found {len(outdated)} outdated packages and {len(vulnerabilities)} vulnerabilities")

    elif args.action == "security":
        success = updater.run_security_updates(args.auto_merge)
        if success:
            print("Security updates completed successfully")
        else:
            print("Security updates failed")
            sys.exit(1)

    elif args.action == "update":
        success = updater.run_regular_updates(args.update_type)
        if success:
            print(f"{args.update_type.title()} updates completed successfully")
        else:
            print(f"{args.update_type.title()} updates failed")
            sys.exit(1)

    elif args.action == "report":
        report = updater.generate_update_report()
        if args.output_report:
            with open(args.output_report, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output_report}")
        else:
            print(report)


if __name__ == "__main__":
    main()