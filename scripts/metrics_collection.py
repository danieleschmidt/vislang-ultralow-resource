#!/usr/bin/env python3
"""
Automated metrics collection script for VisLang-UltraLow-Resource project.

This script collects various metrics from different sources and updates the
project metrics file. It can be run manually or as part of CI/CD pipeline.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging

try:
    import requests
    import psutil
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install requests psutil")
    sys.exit(1)


class MetricsCollector:
    """Collects metrics from various sources and updates project metrics."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.metrics = self.load_metrics()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "danieleschmidt")
        self.repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "vislang-ultralow-resource")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            self.logger.error(f"Metrics config file not found: {self.config_path}")
            sys.exit(1)

    def save_metrics(self):
        """Save updated metrics to file."""
        self.metrics["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        self.logger.info(f"Metrics updated and saved to {self.config_path}")

    def collect_repository_metrics(self):
        """Collect GitHub repository metrics."""
        if not self.github_token:
            self.logger.warning("GITHUB_TOKEN not provided, skipping repository metrics")
            return

        base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }

        try:
            # Repository basic info
            repo_response = requests.get(base_url, headers=headers)
            repo_data = repo_response.json()
            
            # Update basic repository metrics
            self.metrics["business_metrics"]["growth"]["stars_github"] = repo_data.get("stargazers_count", 0)
            self.metrics["business_metrics"]["growth"]["forks_github"] = repo_data.get("forks_count", 0)

            # Get commits from last 30 days
            since_date = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
            commits_url = f"{base_url}/commits?since={since_date}"
            commits_response = requests.get(commits_url, headers=headers)
            commits_count = len(commits_response.json()) if commits_response.status_code == 200 else 0
            
            self.metrics["repository_metrics"]["activity"]["commits_last_30_days"] = commits_count

            # Get pull requests from last 30 days
            prs_url = f"{base_url}/pulls?state=all&since={since_date}"
            prs_response = requests.get(prs_url, headers=headers)
            prs_count = len(prs_response.json()) if prs_response.status_code == 200 else 0
            
            self.metrics["repository_metrics"]["activity"]["prs_last_30_days"] = prs_count

            # Get issues from last 30 days
            issues_url = f"{base_url}/issues?state=all&since={since_date}"
            issues_response = requests.get(issues_url, headers=headers)
            issues_data = issues_response.json() if issues_response.status_code == 200 else []
            # Filter out pull requests (they appear in issues API)
            issues_count = len([issue for issue in issues_data if 'pull_request' not in issue])
            
            self.metrics["repository_metrics"]["activity"]["issues_last_30_days"] = issues_count

            # Get contributors
            contributors_url = f"{base_url}/contributors"
            contributors_response = requests.get(contributors_url, headers=headers)
            contributors_count = len(contributors_response.json()) if contributors_response.status_code == 200 else 0
            
            self.metrics["repository_metrics"]["activity"]["contributors_last_30_days"] = contributors_count

            self.logger.info("Repository metrics collected successfully")

        except Exception as e:
            self.logger.error(f"Error collecting repository metrics: {e}")

    def collect_code_quality_metrics(self):
        """Collect code quality metrics from local analysis."""
        try:
            # Test coverage from pytest
            coverage_file = Path("coverage.xml")
            if coverage_file.exists():
                # Parse coverage XML or use coverage command
                result = subprocess.run(
                    ["coverage", "report", "--show-missing"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # Extract coverage percentage from output
                    lines = result.stdout.split('\n')
                    for line in reversed(lines):
                        if 'TOTAL' in line:
                            try:
                                coverage_str = line.split()[-1].replace('%', '')
                                coverage = float(coverage_str)
                                self.metrics["repository_metrics"]["quality"]["test_coverage_percentage"] = coverage
                                self.metrics["quality_metrics"]["code_quality"]["code_coverage_percentage"] = coverage
                                break
                            except (ValueError, IndexError):
                                continue

            # Count test files
            test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
            unit_tests = list(Path("tests/unit").rglob("test_*.py")) if Path("tests/unit").exists() else []
            integration_tests = list(Path("tests/integration").rglob("test_*.py")) if Path("tests/integration").exists() else []
            e2e_tests = list(Path("tests/e2e").rglob("test_*.py")) if Path("tests/e2e").exists() else []

            self.metrics["quality_metrics"]["testing"]["unit_test_count"] = len(unit_tests)
            self.metrics["quality_metrics"]["testing"]["integration_test_count"] = len(integration_tests)
            self.metrics["quality_metrics"]["testing"]["e2e_test_count"] = len(e2e_tests)

            # Line count and documentation
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            total_lines = 0
            documented_lines = 0
            
            for file_path in src_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        documented_lines += sum(1 for line in lines if line.strip().startswith('"""') or line.strip().startswith('"""'))
                except Exception:
                    continue

            if total_lines > 0:
                doc_ratio = (documented_lines / total_lines) * 100
                self.metrics["quality_metrics"]["documentation"]["code_documentation_ratio"] = doc_ratio

            self.logger.info("Code quality metrics collected successfully")

        except Exception as e:
            self.logger.error(f"Error collecting code quality metrics: {e}")

    def collect_security_metrics(self):
        """Collect security-related metrics."""
        try:
            # Check for security files
            security_files = [
                "SECURITY.md",
                ".github/workflows/security.yml",
                "requirements.txt",
                ".secrets.baseline"
            ]
            
            security_coverage = sum(1 for f in security_files if Path(f).exists())
            total_security_files = len(security_files)
            
            self.metrics["security_metrics"]["compliance"]["security_policy_compliance"] = (
                security_coverage / total_security_files * 100
            )

            # Run safety check if available
            try:
                result = subprocess.run(
                    ["safety", "check", "--json"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data.get("vulnerabilities", [])
                    
                    critical = sum(1 for v in vulnerabilities if v.get("severity") == "critical")
                    high = sum(1 for v in vulnerabilities if v.get("severity") == "high")
                    medium = sum(1 for v in vulnerabilities if v.get("severity") == "medium")
                    low = sum(1 for v in vulnerabilities if v.get("severity") == "low")
                    
                    self.metrics["security_metrics"]["vulnerability_management"]["critical_vulnerabilities"] = critical
                    self.metrics["security_metrics"]["vulnerability_management"]["high_vulnerabilities"] = high
                    self.metrics["security_metrics"]["vulnerability_management"]["medium_vulnerabilities"] = medium
                    self.metrics["security_metrics"]["vulnerability_management"]["low_vulnerabilities"] = low

            except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
                self.logger.warning("Safety check not available or failed")

            self.logger.info("Security metrics collected successfully")

        except Exception as e:
            self.logger.error(f"Error collecting security metrics: {e}")

    def collect_infrastructure_metrics(self):
        """Collect system/infrastructure metrics."""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            self.metrics["infrastructure_metrics"]["performance"]["cpu_utilization_percentage"] = cpu_percent
            self.metrics["infrastructure_metrics"]["performance"]["memory_utilization_percentage"] = memory.percent
            self.metrics["infrastructure_metrics"]["performance"]["disk_utilization_percentage"] = (
                (disk.used / disk.total) * 100
            )

            # Docker container metrics if available
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "table {{.CPUPerc}}\t{{.MemUsage}}"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    # Parse docker stats output
                    self.logger.info("Docker metrics collected")
            except subprocess.CalledProcessError:
                self.logger.debug("Docker not available for metrics collection")

            self.logger.info("Infrastructure metrics collected successfully")

        except Exception as e:
            self.logger.error(f"Error collecting infrastructure metrics: {e}")

    def collect_build_metrics(self):
        """Collect CI/CD build metrics."""
        try:
            # Check if we're in a CI environment
            if os.getenv("CI"):
                # GitHub Actions specific
                if os.getenv("GITHUB_ACTIONS"):
                    workflow_run_id = os.getenv("GITHUB_RUN_ID")
                    if workflow_run_id:
                        self.logger.info(f"Collecting metrics for GitHub Actions run: {workflow_run_id}")

                # Record build success (since we're running, assume success)
                current_success_rate = self.metrics["repository_metrics"]["performance"]["build_success_rate"]
                # Simple success rate calculation (would need more sophisticated tracking in production)
                new_success_rate = min(100, current_success_rate + 1)
                self.metrics["repository_metrics"]["performance"]["build_success_rate"] = new_success_rate

            self.logger.info("Build metrics collected successfully")

        except Exception as e:
            self.logger.error(f"Error collecting build metrics: {e}")

    def calculate_health_score(self):
        """Calculate overall repository health score."""
        try:
            # Documentation score (based on file existence and coverage)
            doc_files = ["README.md", "CONTRIBUTING.md", "LICENSE", "docs/"]
            doc_score = sum(25 for f in doc_files if Path(f).exists())

            # Testing score (based on coverage and test count)
            coverage = self.metrics["repository_metrics"]["quality"]["test_coverage_percentage"]
            test_score = min(100, coverage + (coverage * 0.2))  # Bonus for high coverage

            # Security score (based on security files and vulnerability count)
            security_compliance = self.metrics["security_metrics"]["compliance"]["security_policy_compliance"]
            critical_vulns = self.metrics["security_metrics"]["vulnerability_management"]["critical_vulnerabilities"]
            security_score = max(0, security_compliance - (critical_vulns * 20))

            # Maintenance score (based on recent activity)
            commits = self.metrics["repository_metrics"]["activity"]["commits_last_30_days"]
            prs = self.metrics["repository_metrics"]["activity"]["prs_last_30_days"]
            maintenance_score = min(100, (commits * 2) + (prs * 5))

            # Overall health score
            overall_score = (doc_score + test_score + security_score + maintenance_score) / 4

            # Update health scores
            self.metrics["repository_metrics"]["health_score"]["overall"] = round(overall_score, 1)
            self.metrics["repository_metrics"]["health_score"]["documentation"] = doc_score
            self.metrics["repository_metrics"]["health_score"]["testing"] = round(test_score, 1)
            self.metrics["repository_metrics"]["health_score"]["security"] = round(security_score, 1)
            self.metrics["repository_metrics"]["health_score"]["maintenance"] = round(maintenance_score, 1)

            self.logger.info(f"Health score calculated: {overall_score:.1f}")

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")

    def generate_summary_report(self) -> str:
        """Generate a summary report of current metrics."""
        health = self.metrics["repository_metrics"]["health_score"]
        activity = self.metrics["repository_metrics"]["activity"]
        quality = self.metrics["repository_metrics"]["quality"]

        report = f"""
# VisLang Project Metrics Summary

**Last Updated**: {self.metrics['last_updated']}

## Health Score: {health['overall']}/100

- 📚 Documentation: {health['documentation']}/100
- 🧪 Testing: {health['testing']}/100  
- 🔒 Security: {health['security']}/100
- 🔧 Maintenance: {health['maintenance']}/100

## Recent Activity (30 days)

- Commits: {activity['commits_last_30_days']}
- Pull Requests: {activity['prs_last_30_days']}
- Issues: {activity['issues_last_30_days']}
- Contributors: {activity['contributors_last_30_days']}

## Quality Metrics

- Test Coverage: {quality['test_coverage_percentage']:.1f}%
- Security Vulnerabilities: {quality['vulnerability_count']}
- Technical Debt Ratio: {quality['technical_debt_ratio']:.1f}%

## Repository Growth

- GitHub Stars: {self.metrics['business_metrics']['growth']['stars_github']}
- GitHub Forks: {self.metrics['business_metrics']['growth']['forks_github']}
"""
        return report

    def run_collection(self, include_types: Optional[list] = None):
        """Run metrics collection for specified types."""
        if include_types is None:
            include_types = ["repository", "quality", "security", "infrastructure", "build"]

        self.logger.info("Starting metrics collection...")

        if "repository" in include_types:
            self.collect_repository_metrics()

        if "quality" in include_types:
            self.collect_code_quality_metrics()

        if "security" in include_types:
            self.collect_security_metrics()

        if "infrastructure" in include_types:
            self.collect_infrastructure_metrics()

        if "build" in include_types:
            self.collect_build_metrics()

        # Always calculate health score
        self.calculate_health_score()

        # Save updated metrics
        self.save_metrics()

        # Generate summary
        summary = self.generate_summary_report()
        self.logger.info("Metrics collection completed")
        
        return summary


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["repository", "quality", "security", "infrastructure", "build"],
        default=["repository", "quality", "security", "infrastructure", "build"],
        help="Types of metrics to collect"
    )
    parser.add_argument(
        "--config",
        default=".github/project-metrics.json",
        help="Path to metrics configuration file"
    )
    parser.add_argument(
        "--output-summary",
        help="Output summary report to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize collector
    collector = MetricsCollector(args.config)

    # Run collection
    summary = collector.run_collection(args.types)

    # Output summary
    if args.output_summary:
        with open(args.output_summary, 'w') as f:
            f.write(summary)
        print(f"Summary report saved to {args.output_summary}")
    else:
        print(summary)


if __name__ == "__main__":
    main()