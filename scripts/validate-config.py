#!/usr/bin/env python3
"""
Configuration validation script for VisLang-UltraLow-Resource project.

This script validates all configuration files and environment settings
to ensure proper integration of all components.
"""

import os
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import re
import subprocess
import tempfile

try:
    import toml
    import docker
    import requests
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install toml docker requests")
    sys.exit(1)


class ConfigValidator:
    """Validates project configuration and component integration."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.errors = []
        self.warnings = []
        self.info = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_project_structure(self) -> bool:
        """Validate project directory structure."""
        self.logger.info("Validating project structure...")
        
        required_files = [
            "pyproject.toml",
            "README.md",
            ".gitignore",
            "Dockerfile"
        ]
        
        required_directories = [
            "src",
            "tests", 
            "scripts",
            "docs"
        ]
        
        success = True
        
        # Check required files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.errors.append(f"Missing required file: {file_path}")
                success = False
            else:
                self.info.append(f"Found required file: {file_path}")
        
        # Check required directories
        for dir_path in required_directories:
            full_path = self.project_root / dir_path
            if not full_path.is_dir():
                self.errors.append(f"Missing required directory: {dir_path}")
                success = False
            else:
                self.info.append(f"Found required directory: {dir_path}")
        
        return success

    def validate_python_configuration(self) -> bool:
        """Validate Python project configuration."""
        self.logger.info("Validating Python configuration...")
        
        success = True
        pyproject_file = self.project_root / "pyproject.toml"
        
        if not pyproject_file.exists():
            self.errors.append("pyproject.toml not found")
            return False
        
        try:
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            # Check required project metadata
            required_fields = ['name', 'version', 'description']
            project_section = pyproject_data.get('project', {})
            
            for field in required_fields:
                if field not in project_section:
                    self.errors.append(f"Missing required field in pyproject.toml: project.{field}")
                    success = False
            
            # Check dependencies
            if 'dependencies' in project_section:
                deps = project_section['dependencies']
                self.info.append(f"Found {len(deps)} main dependencies")
                
                # Check for common ML dependencies
                ml_deps = ['torch', 'transformers', 'datasets', 'numpy', 'pandas']
                found_ml_deps = [dep for dep in deps if any(ml_dep in dep.lower() for ml_dep in ml_deps)]
                self.info.append(f"Found {len(found_ml_deps)} ML-related dependencies")
            
            # Check optional dependencies
            if 'optional-dependencies' in project_section:
                optional_deps = project_section['optional-dependencies']
                self.info.append(f"Found {len(optional_deps)} optional dependency groups")
            
            # Check build system
            if 'build-system' not in pyproject_data:
                self.warnings.append("No build-system configuration found")
            else:
                build_system = pyproject_data['build-system']
                if 'requires' not in build_system:
                    self.warnings.append("No build-system.requires found")
                if 'build-backend' not in build_system:
                    self.warnings.append("No build-system.build-backend found")
        
        except Exception as e:
            self.errors.append(f"Failed to parse pyproject.toml: {e}")
            success = False
        
        return success

    def validate_environment_variables(self) -> bool:
        """Validate required environment variables."""
        self.logger.info("Validating environment variables...")
        
        # Check for .env.example file
        env_example = self.project_root / ".env.example"
        if env_example.exists():
            self.info.append("Found .env.example file")
        else:
            self.warnings.append("No .env.example file found")
        
        # Required environment variables for development
        required_env_vars = [
            'DEBUG',
            'LOG_LEVEL',
            'SECRET_KEY'
        ]
        
        # Optional but recommended environment variables
        recommended_env_vars = [
            'DATABASE_URL',
            'REDIS_URL', 
            'API_HOST',
            'API_PORT',
            'PROMETHEUS_PORT',
            'GRAFANA_PORT'
        ]
        
        success = True
        
        # Check required variables
        for var in required_env_vars:
            if var not in os.environ:
                self.warnings.append(f"Recommended environment variable not set: {var}")
        
        # Check recommended variables
        for var in recommended_env_vars:
            if var in os.environ:
                self.info.append(f"Found optional environment variable: {var}")
        
        return success

    def validate_docker_configuration(self) -> bool:
        """Validate Docker configuration."""
        self.logger.info("Validating Docker configuration...")
        
        success = True
        
        # Check Dockerfile
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            self.errors.append("Dockerfile not found")
            return False
        
        try:
            with open(dockerfile, 'r') as f:
                dockerfile_content = f.read()
            
            # Check for multi-stage build
            if 'FROM' in dockerfile_content:
                from_count = dockerfile_content.count('FROM')
                if from_count > 1:
                    self.info.append(f"Multi-stage Docker build detected ({from_count} stages)")
                else:
                    self.warnings.append("Single-stage Docker build (consider multi-stage for optimization)")
            
            # Check for security best practices
            security_patterns = [
                (r'USER\s+\w+', "Non-root user configuration found"),
                (r'COPY.*--chown=', "File ownership configuration found"),
                (r'RUN.*apt-get update.*apt-get install', "Package installation found")
            ]
            
            for pattern, message in security_patterns:
                if re.search(pattern, dockerfile_content, re.IGNORECASE):
                    self.info.append(message)
        
        except Exception as e:
            self.errors.append(f"Failed to parse Dockerfile: {e}")
            success = False
        
        # Check docker-compose files
        compose_files = [
            "docker-compose.yml",
            "docker-compose.dev.yml",
            "docker-compose.prod.yml"
        ]
        
        for compose_file in compose_files:
            compose_path = self.project_root / compose_file
            if compose_path.exists():
                self.info.append(f"Found Docker Compose file: {compose_file}")
                
                try:
                    with open(compose_path, 'r') as f:
                        compose_data = yaml.safe_load(f)
                    
                    if 'services' in compose_data:
                        services = list(compose_data['services'].keys())
                        self.info.append(f"Docker Compose services in {compose_file}: {', '.join(services)}")
                
                except Exception as e:
                    self.warnings.append(f"Failed to parse {compose_file}: {e}")
        
        # Check .dockerignore
        dockerignore = self.project_root / ".dockerignore"
        if dockerignore.exists():
            self.info.append("Found .dockerignore file")
        else:
            self.warnings.append("No .dockerignore file (recommended for build optimization)")
        
        return success

    def validate_testing_configuration(self) -> bool:
        """Validate testing configuration."""
        self.logger.info("Validating testing configuration...")
        
        success = True
        
        # Check test directories
        test_dirs = ["tests", "test"]
        test_dir = None
        
        for test_dirname in test_dirs:
            test_path = self.project_root / test_dirname
            if test_path.is_dir():
                test_dir = test_path
                self.info.append(f"Found test directory: {test_dirname}")
                break
        
        if not test_dir:
            self.errors.append("No test directory found")
            return False
        
        # Check for test configuration files
        test_config_files = [
            "pytest.ini",
            "pyproject.toml",  # pytest configuration can be in pyproject.toml
            "setup.cfg",
            ".coveragerc",
            "coverage.ini"
        ]
        
        pytest_config_found = False
        for config_file in test_config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                if config_file in ["pytest.ini", "pyproject.toml", "setup.cfg"]:
                    pytest_config_found = True
                self.info.append(f"Found test configuration: {config_file}")
        
        if not pytest_config_found:
            self.warnings.append("No pytest configuration found")
        
        # Check for test files
        test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
        if test_files:
            self.info.append(f"Found {len(test_files)} test files")
        else:
            self.warnings.append("No test files found")
        
        # Check test structure
        test_subdirs = ["unit", "integration", "e2e", "performance"]
        for subdir in test_subdirs:
            subdir_path = test_dir / subdir
            if subdir_path.is_dir():
                self.info.append(f"Found test category: {subdir}")
        
        return success

    def validate_monitoring_configuration(self) -> bool:
        """Validate monitoring and observability configuration."""
        self.logger.info("Validating monitoring configuration...")
        
        success = True
        
        # Check monitoring directory
        monitoring_dir = self.project_root / "monitoring"
        if not monitoring_dir.is_dir():
            self.warnings.append("No monitoring directory found")
            return True  # Not critical
        
        # Check monitoring configuration files
        monitoring_files = [
            "prometheus.yml",
            "grafana/dashboard.json",
            "alertmanager.yml",
            "docker-compose.yml"
        ]
        
        for file_path in monitoring_files:
            full_path = monitoring_dir / file_path
            if full_path.exists():
                self.info.append(f"Found monitoring config: {file_path}")
            else:
                self.warnings.append(f"Missing monitoring config: {file_path}")
        
        # Check health check script
        health_check_script = self.project_root / "scripts" / "monitoring" / "health-check.py"
        if health_check_script.exists():
            self.info.append("Found health check script")
        else:
            self.warnings.append("No health check script found")
        
        return success

    def validate_automation_configuration(self) -> bool:
        """Validate automation configuration."""
        self.logger.info("Validating automation configuration...")
        
        success = True
        
        # Check automation scripts
        automation_dir = self.project_root / "scripts" / "automation"
        if automation_dir.is_dir():
            self.info.append("Found automation directory")
            
            automation_scripts = [
                "dependency-updater.py",
                "release-automation.py",
                "automation-config.yaml"
            ]
            
            for script in automation_scripts:
                script_path = automation_dir / script
                if script_path.exists():
                    self.info.append(f"Found automation script: {script}")
                else:
                    self.warnings.append(f"Missing automation script: {script}")
        else:
            self.warnings.append("No automation directory found")
        
        # Check metrics collection
        metrics_script = self.project_root / "scripts" / "metrics_collection.py"
        if metrics_script.exists():
            self.info.append("Found metrics collection script")
        else:
            self.warnings.append("No metrics collection script found")
        
        # Check metrics configuration
        metrics_config = self.project_root / ".github" / "project-metrics.json"
        if metrics_config.exists():
            self.info.append("Found metrics configuration")
            
            try:
                with open(metrics_config, 'r') as f:
                    metrics_data = json.load(f)
                
                required_sections = [
                    'repository_metrics',
                    'quality_metrics', 
                    'security_metrics',
                    'infrastructure_metrics'
                ]
                
                for section in required_sections:
                    if section in metrics_data:
                        self.info.append(f"Found metrics section: {section}")
                    else:
                        self.warnings.append(f"Missing metrics section: {section}")
            
            except Exception as e:
                self.warnings.append(f"Failed to parse metrics configuration: {e}")
        else:
            self.warnings.append("No metrics configuration found")
        
        return success

    def validate_workflow_configuration(self) -> bool:
        """Validate GitHub Actions workflow configuration."""
        self.logger.info("Validating GitHub Actions configuration...")
        
        success = True
        
        # Check .github directory
        github_dir = self.project_root / ".github"
        if not github_dir.is_dir():
            self.warnings.append("No .github directory found")
            return True
        
        # Check workflows directory
        workflows_dir = github_dir / "workflows"
        if not workflows_dir.is_dir():
            self.warnings.append("No .github/workflows directory found")
        else:
            workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
            if workflow_files:
                self.info.append(f"Found {len(workflow_files)} workflow files")
                for workflow_file in workflow_files:
                    self.info.append(f"  - {workflow_file.name}")
            else:
                self.warnings.append("No workflow files found")
        
        # Check workflow examples
        examples_dir = self.project_root / "docs" / "workflows" / "examples"
        if examples_dir.is_dir():
            example_files = list(examples_dir.glob("*.yml"))
            if example_files:
                self.info.append(f"Found {len(example_files)} workflow examples")
        
        # Check dependabot configuration
        dependabot_config = github_dir / "dependabot.yml"
        if dependabot_config.exists():
            self.info.append("Found Dependabot configuration")
        else:
            self.warnings.append("No Dependabot configuration found")
        
        return success

    def validate_security_configuration(self) -> bool:
        """Validate security configuration."""
        self.logger.info("Validating security configuration...")
        
        success = True
        
        # Check security policy
        security_policy = self.project_root / "SECURITY.md"
        if security_policy.exists():
            self.info.append("Found security policy (SECURITY.md)")
        else:
            self.warnings.append("No security policy found")
        
        # Check secrets baseline
        secrets_baseline = self.project_root / ".secrets.baseline"
        if secrets_baseline.exists():
            self.info.append("Found secrets baseline")
        else:
            self.warnings.append("No secrets baseline found")
        
        # Check security configuration files
        security_configs = [
            ".bandit",
            ".safety-policy.json",
            "trivy.yaml"
        ]
        
        for config in security_configs:
            config_path = self.project_root / config
            if config_path.exists():
                self.info.append(f"Found security config: {config}")
            else:
                self.warnings.append(f"Security config not found: {config}")
        
        return success

    def validate_dependencies(self) -> bool:
        """Validate project dependencies."""
        self.logger.info("Validating dependencies...")
        
        success = True
        
        # Check if virtual environment is active
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.info.append("Running in virtual environment")
        else:
            self.warnings.append("Not running in virtual environment (recommended)")
        
        # Try to import key dependencies
        key_dependencies = [
            'toml',
            'yaml',
            'requests',
            'docker'
        ]
        
        for dep in key_dependencies:
            try:
                __import__(dep)
                self.info.append(f"Successfully imported: {dep}")
            except ImportError:
                self.warnings.append(f"Failed to import: {dep}")
        
        return success

    def validate_external_services(self) -> bool:
        """Validate connectivity to external services."""
        self.logger.info("Validating external services...")
        
        success = True
        
        # Test services based on environment variables
        services_to_test = []
        
        if 'DATABASE_URL' in os.environ:
            services_to_test.append(('Database', os.environ['DATABASE_URL']))
        
        if 'REDIS_URL' in os.environ:
            services_to_test.append(('Redis', os.environ['REDIS_URL']))
        
        # Test Docker daemon
        try:
            client = docker.from_env()
            client.ping()
            self.info.append("Docker daemon is accessible")
        except Exception as e:
            self.warnings.append(f"Docker daemon not accessible: {e}")
        
        # Test internet connectivity (for dependency updates, etc.)
        try:
            response = requests.get('https://pypi.org', timeout=5)
            if response.status_code == 200:
                self.info.append("Internet connectivity confirmed")
            else:
                self.warnings.append("Limited internet connectivity")
        except Exception as e:
            self.warnings.append(f"Internet connectivity issue: {e}")
        
        return success

    def run_validation(self, include_external: bool = True) -> Dict[str, Any]:
        """Run all validation checks."""
        self.logger.info("Starting configuration validation...")
        
        validation_results = {}
        
        # Run validation checks
        checks = [
            ('project_structure', self.validate_project_structure),
            ('python_config', self.validate_python_configuration),
            ('environment_vars', self.validate_environment_variables),
            ('docker_config', self.validate_docker_configuration),
            ('testing_config', self.validate_testing_configuration),
            ('monitoring_config', self.validate_monitoring_configuration),
            ('automation_config', self.validate_automation_configuration),
            ('workflow_config', self.validate_workflow_configuration),
            ('security_config', self.validate_security_configuration),
            ('dependencies', self.validate_dependencies)
        ]
        
        if include_external:
            checks.append(('external_services', self.validate_external_services))
        
        for check_name, check_function in checks:
            try:
                result = check_function()
                validation_results[check_name] = {
                    'success': result,
                    'timestamp': None  # Could add timestamp if needed
                }
            except Exception as e:
                self.logger.error(f"Validation check '{check_name}' failed: {e}")
                validation_results[check_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Compile results
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        total_info = len(self.info)
        
        overall_success = total_errors == 0
        
        results = {
            'success': overall_success,
            'summary': {
                'errors': total_errors,
                'warnings': total_warnings,
                'info': total_info
            },
            'details': {
                'errors': self.errors,
                'warnings': self.warnings,
                'info': self.info
            },
            'validation_results': validation_results
        }
        
        self.logger.info(f"Validation completed: {total_errors} errors, {total_warnings} warnings, {total_info} info")
        
        return results

    def print_results(self, results: Dict[str, Any]):
        """Print validation results in a formatted way."""
        
        print("\n" + "="*60)
        print("VISLANG PROJECT CONFIGURATION VALIDATION RESULTS")
        print("="*60)
        
        summary = results['summary']
        
        # Overall status
        if results['success']:
            print("\n✅ VALIDATION PASSED")
        else:
            print("\n❌ VALIDATION FAILED")
        
        print(f"\nSummary:")
        print(f"  Errors:   {summary['errors']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Info:     {summary['info']}")
        
        details = results['details']
        
        # Print errors
        if details['errors']:
            print(f"\n🚨 ERRORS ({len(details['errors'])}):")
            for i, error in enumerate(details['errors'], 1):
                print(f"  {i}. {error}")
        
        # Print warnings
        if details['warnings']:
            print(f"\n⚠️  WARNINGS ({len(details['warnings'])}):")
            for i, warning in enumerate(details['warnings'], 1):
                print(f"  {i}. {warning}")
        
        # Print info (only first 10 to avoid spam)
        if details['info']:
            info_to_show = details['info'][:10]
            print(f"\nℹ️  INFO ({len(info_to_show)} of {len(details['info'])}):")
            for i, info in enumerate(info_to_show, 1):
                print(f"  {i}. {info}")
            
            if len(details['info']) > 10:
                print(f"  ... and {len(details['info']) - 10} more")
        
        # Print validation results summary
        print(f"\n📋 VALIDATION CHECKS:")
        for check_name, check_result in results['validation_results'].items():
            status = "✅" if check_result['success'] else "❌"
            readable_name = check_name.replace('_', ' ').title()
            print(f"  {status} {readable_name}")
        
        print("\n" + "="*60)


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Validate VisLang project configuration")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project"
    )
    parser.add_argument(
        "--no-external",
        action="store_true",
        help="Skip external service validation"
    )
    parser.add_argument(
        "--json",
        help="Save results as JSON to specified file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize validator
    validator = ConfigValidator(args.project_root)

    # Run validation
    results = validator.run_validation(include_external=not args.no_external)

    # Print results
    validator.print_results(results)

    # Save JSON results if requested
    if args.json:
        try:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.json}")
        except Exception as e:
            print(f"Failed to save JSON results: {e}")

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()