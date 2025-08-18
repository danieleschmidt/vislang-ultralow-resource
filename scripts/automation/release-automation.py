#!/usr/bin/env python3
"""
Automated release management script for VisLang-UltraLow-Resource project.

This script handles version bumping, changelog generation, and release preparation.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import re
import tempfile

try:
    import toml
    import requests
    from semantic_version import Version
    import git
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install toml requests PyGithub GitPython semantic_version")
    sys.exit(1)


class ReleaseAutomation:
    """Automated release management for the VisLang project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Git repository
        try:
            self.repo = git.Repo(self.project_root)
        except git.InvalidGitRepositoryError:
            self.logger.error("Not a valid Git repository")
            sys.exit(1)
            
        # GitHub configuration
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "danieleschmidt")
        self.repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "vislang-ultralow-resource")

    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        try:
            with open(self.pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            return pyproject_data.get('project', {}).get('version', '0.0.0')
            
        except Exception as e:
            self.logger.error(f"Failed to read current version: {e}")
            return '0.0.0'

    def get_latest_tag(self) -> Optional[str]:
        """Get the latest Git tag."""
        try:
            # Get all tags sorted by version
            tags = []
            for tag in self.repo.tags:
                try:
                    # Only consider version tags (v1.0.0 format)
                    if tag.name.startswith('v'):
                        version_str = tag.name[1:]  # Remove 'v' prefix
                        Version(version_str)  # Validate version format
                        tags.append((tag.name, tag.commit.committed_date))
                except:
                    continue
            
            if not tags:
                return None
                
            # Sort by commit date and return the latest
            tags.sort(key=lambda x: x[1], reverse=True)
            return tags[0][0]
            
        except Exception as e:
            self.logger.error(f"Failed to get latest tag: {e}")
            return None

    def bump_version(self, version: str, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        try:
            current_version = Version(version)
            
            if bump_type == 'major':
                new_version = current_version.next_major()
            elif bump_type == 'minor':
                new_version = current_version.next_minor()
            elif bump_type == 'patch':
                new_version = current_version.next_patch()
            else:
                raise ValueError(f"Invalid bump type: {bump_type}")
                
            return str(new_version)
            
        except Exception as e:
            self.logger.error(f"Failed to bump version: {e}")
            raise

    def update_version_in_files(self, new_version: str) -> bool:
        """Update version in project files."""
        success = True
        
        # Update pyproject.toml
        try:
            with open(self.pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
                
            pyproject_data['project']['version'] = new_version
            
            with open(self.pyproject_file, 'w') as f:
                toml.dump(pyproject_data, f)
                
            self.logger.info(f"Updated version in pyproject.toml to {new_version}")
            
        except Exception as e:
            self.logger.error(f"Failed to update pyproject.toml: {e}")
            success = False
            
        # Update version in Python package __init__.py if it exists
        init_files = [
            self.project_root / "src" / "vislang_ultralow" / "__init__.py",
            self.project_root / "vislang_ultralow" / "__init__.py"
        ]
        
        for init_file in init_files:
            if init_file.exists():
                try:
                    with open(init_file, 'r') as f:
                        content = f.read()
                    
                    # Update __version__ variable
                    new_content = re.sub(
                        r'__version__\s*=\s*["\'].*?["\']',
                        f'__version__ = "{new_version}"',
                        content
                    )
                    
                    with open(init_file, 'w') as f:
                        f.write(new_content)
                        
                    self.logger.info(f"Updated version in {init_file}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to update {init_file}: {e}")
                    
        return success

    def get_commits_since_tag(self, tag: Optional[str] = None) -> List[Dict[str, str]]:
        """Get commits since the last tag."""
        try:
            if tag:
                # Get commits since specific tag
                commit_range = f"{tag}..HEAD"
            else:
                # Get all commits if no tag exists
                commit_range = "HEAD"
                
            commits = list(self.repo.iter_commits(commit_range))
            
            commit_list = []
            for commit in commits:
                commit_list.append({
                    'hash': commit.hexsha[:8],
                    'message': commit.message.strip(),
                    'author': str(commit.author),
                    'date': datetime.fromtimestamp(commit.committed_date, tz=timezone.utc).isoformat(),
                    'files': [item.a_path for item in commit.stats.files.keys()]
                })
                
            return commit_list
            
        except Exception as e:
            self.logger.error(f"Failed to get commits: {e}")
            return []

    def categorize_commits(self, commits: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
        """Categorize commits by type (feat, fix, docs, etc.)."""
        categories = {
            'features': [],
            'fixes': [],
            'docs': [],
            'tests': [],
            'refactor': [],
            'perf': [],
            'build': [],
            'ci': [],
            'style': [],
            'chore': [],
            'other': []
        }
        
        # Patterns for categorizing commits
        patterns = {
            'features': [r'^feat(\(.*\))?:', r'^feature:', r'^add:'],
            'fixes': [r'^fix(\(.*\))?:', r'^bugfix:', r'^bug:'],
            'docs': [r'^docs(\(.*\))?:', r'^doc:', r'^documentation:'],
            'tests': [r'^test(\(.*\))?:', r'^tests:', r'^testing:'],
            'refactor': [r'^refactor(\(.*\))?:', r'^refactoring:'],
            'perf': [r'^perf(\(.*\))?:', r'^performance:'],
            'build': [r'^build(\(.*\))?:', r'^deps:', r'^dependencies:'],
            'ci': [r'^ci(\(.*\))?:', r'^workflow:', r'^github:'],
            'style': [r'^style(\(.*\))?:', r'^formatting:'],
            'chore': [r'^chore(\(.*\))?:', r'^cleanup:', r'^misc:']
        }
        
        for commit in commits:
            message = commit['message'].lower()
            categorized = False
            
            for category, category_patterns in patterns.items():
                for pattern in category_patterns:
                    if re.match(pattern, message):
                        categories[category].append(commit)
                        categorized = True
                        break
                if categorized:
                    break
                    
            if not categorized:
                categories['other'].append(commit)
                
        return categories

    def generate_changelog_entry(self, version: str, commits: List[Dict[str, str]]) -> str:
        """Generate changelog entry for the new version."""
        categorized_commits = self.categorize_commits(commits)
        
        changelog_entry = f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        # Only include categories that have commits
        category_titles = {
            'features': '### 🚀 Features',
            'fixes': '### 🐛 Bug Fixes',
            'docs': '### 📚 Documentation',
            'tests': '### 🧪 Tests',
            'refactor': '### ♻️ Refactoring',
            'perf': '### ⚡ Performance',
            'build': '### 🔧 Build System',
            'ci': '### 👷 CI/CD',
            'style': '### 💄 Style',
            'chore': '### 🔧 Chore',
            'other': '### 📦 Other'
        }
        
        for category, title in category_titles.items():
            if categorized_commits[category]:
                changelog_entry += f"{title}\n\n"
                for commit in categorized_commits[category]:
                    # Clean up commit message
                    message = commit['message'].split('\n')[0]  # First line only
                    message = re.sub(r'^(feat|fix|docs|test|refactor|perf|build|ci|style|chore)(\(.*\))?:\s*', '', message, flags=re.IGNORECASE)
                    
                    changelog_entry += f"- {message} ({commit['hash']})\n"
                changelog_entry += "\n"
        
        return changelog_entry

    def update_changelog(self, new_entry: str) -> bool:
        """Update the CHANGELOG.md file with new entry."""
        try:
            if self.changelog_file.exists():
                with open(self.changelog_file, 'r') as f:
                    existing_content = f.read()
            else:
                existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
            
            # Insert new entry after the header
            lines = existing_content.split('\n')
            header_end = 0
            
            # Find the end of the header (first occurrence of ## or after first few lines)
            for i, line in enumerate(lines):
                if line.startswith('## ') and i > 3:  # Skip title and description
                    header_end = i
                    break
                elif i > 10:  # If no existing releases, insert after header
                    header_end = i
                    break
            
            # Insert new entry
            lines.insert(header_end, new_entry.rstrip())
            
            new_content = '\n'.join(lines)
            
            with open(self.changelog_file, 'w') as f:
                f.write(new_content)
                
            self.logger.info(f"Updated {self.changelog_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update changelog: {e}")
            return False

    def create_git_tag(self, version: str, message: str = None) -> bool:
        """Create and push a Git tag for the release."""
        try:
            tag_name = f"v{version}"
            tag_message = message or f"Release {version}"
            
            # Create annotated tag
            self.repo.create_tag(tag_name, message=tag_message)
            
            # Push tag to origin
            origin = self.repo.remote('origin')
            origin.push(tag_name)
            
            self.logger.info(f"Created and pushed tag {tag_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create Git tag: {e}")
            return False

    def create_github_release(self, version: str, changelog_entry: str) -> bool:
        """Create a GitHub release."""
        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not provided, skipping GitHub release creation")
            return False
            
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Prepare release data
        tag_name = f"v{version}"
        release_name = f"Release {version}"
        
        # Clean up changelog entry for release notes
        release_notes = changelog_entry.replace(f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}", "").strip()
        
        release_data = {
            "tag_name": tag_name,
            "target_commitish": "main",
            "name": release_name,
            "body": release_notes,
            "draft": False,
            "prerelease": self._is_prerelease(version),
            "generate_release_notes": False  # We provide our own
        }
        
        try:
            url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/releases"
            response = requests.post(url, headers=headers, json=release_data, timeout=30)
            
            if response.status_code == 201:
                release_url = response.json().get('html_url', '')
                self.logger.info(f"Created GitHub release: {release_url}")
                return True
            else:
                self.logger.error(f"Failed to create GitHub release: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error creating GitHub release: {e}")
            
        return False

    def _is_prerelease(self, version: str) -> bool:
        """Check if version is a prerelease."""
        return bool(re.search(r'-(alpha|beta|rc|dev)', version))

    def run_tests(self) -> bool:
        """Run the test suite before release."""
        try:
            self.logger.info("Running test suite...")
            
            # Try different test commands
            test_commands = [
                ["python", "-m", "pytest", "--tb=short"],
                ["pytest", "--tb=short"],
                ["python", "-m", "pytest"],
                ["pytest"]
            ]
            
            for cmd in test_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        self.logger.info("Tests passed successfully")
                        return True
                    else:
                        self.logger.warning(f"Tests failed with command {' '.join(cmd)}: {result.stderr}")
                        
                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    self.logger.error("Tests timed out")
                    return False
                    
            self.logger.error("Could not run tests - no suitable test runner found")
            return False
            
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return False

    def create_release(self, bump_type: str, dry_run: bool = False, skip_tests: bool = False) -> Dict[str, Any]:
        """Create a complete release."""
        result = {
            'success': False,
            'version': None,
            'actions_performed': [],
            'errors': []
        }
        
        try:
            # Get current version and calculate new version
            current_version = self.get_current_version()
            new_version = self.bump_version(current_version, bump_type)
            result['version'] = new_version
            
            self.logger.info(f"Creating release: {current_version} -> {new_version}")
            
            if dry_run:
                self.logger.info("DRY RUN MODE - No changes will be made")
            
            # Run tests first
            if not skip_tests and not dry_run:
                if not self.run_tests():
                    result['errors'].append("Tests failed")
                    return result
                result['actions_performed'].append("Tests passed")
            
            # Get commits since last tag
            latest_tag = self.get_latest_tag()
            commits = self.get_commits_since_tag(latest_tag)
            
            if not commits:
                self.logger.warning("No commits since last release")
                result['errors'].append("No commits since last release")
                return result
            
            # Generate changelog entry
            changelog_entry = self.generate_changelog_entry(new_version, commits)
            
            if not dry_run:
                # Update version in files
                if not self.update_version_in_files(new_version):
                    result['errors'].append("Failed to update version in files")
                    return result
                result['actions_performed'].append("Updated version in files")
                
                # Update changelog
                if not self.update_changelog(changelog_entry):
                    result['errors'].append("Failed to update changelog")
                    return result
                result['actions_performed'].append("Updated changelog")
                
                # Commit changes
                self.repo.index.add([str(self.pyproject_file), str(self.changelog_file)])
                
                # Add __init__.py files if they were updated
                init_files = [
                    "src/vislang_ultralow/__init__.py",
                    "vislang_ultralow/__init__.py"
                ]
                for init_file in init_files:
                    if (self.project_root / init_file).exists():
                        self.repo.index.add([init_file])
                
                commit_message = f"chore(release): bump version to {new_version}\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
                self.repo.index.commit(commit_message)
                result['actions_performed'].append("Committed version changes")
                
                # Create and push tag
                if not self.create_git_tag(new_version, f"Release {new_version}"):
                    result['errors'].append("Failed to create Git tag")
                    return result
                result['actions_performed'].append("Created Git tag")
                
                # Push commits
                origin = self.repo.remote('origin')
                origin.push()
                result['actions_performed'].append("Pushed commits")
                
                # Create GitHub release
                if self.create_github_release(new_version, changelog_entry):
                    result['actions_performed'].append("Created GitHub release")
                else:
                    result['errors'].append("Failed to create GitHub release")
            
            else:
                self.logger.info(f"Would create release {new_version} with {len(commits)} commits")
                self.logger.info("Changelog entry:")
                print(changelog_entry)
            
            result['success'] = len(result['errors']) == 0
            
        except Exception as e:
            self.logger.error(f"Release creation failed: {e}")
            result['errors'].append(str(e))
            
        return result


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Automate release creation")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Type of version bump"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests before release"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize release automation
    release_automation = ReleaseAutomation(args.project_root)

    # Create release
    result = release_automation.create_release(
        bump_type=args.bump_type,
        dry_run=args.dry_run,
        skip_tests=args.skip_tests
    )

    # Print results
    if result['success']:
        print(f"✅ Release {result['version']} created successfully!")
        print("Actions performed:")
        for action in result['actions_performed']:
            print(f"  - {action}")
    else:
        print(f"❌ Release creation failed!")
        print("Errors:")
        for error in result['errors']:
            print(f"  - {error}")
            
        if result['actions_performed']:
            print("Actions that were completed:")
            for action in result['actions_performed']:
                print(f"  - {action}")

    # Exit with appropriate code
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()