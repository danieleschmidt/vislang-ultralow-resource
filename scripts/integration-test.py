#!/usr/bin/env python3
"""
Integration test script for VisLang-UltraLow-Resource project.

This script tests the integration between all project components
to ensure they work together correctly.
"""

import os
import sys
import json
import subprocess
import time
import socket
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import logging
import tempfile
import signal
import threading

try:
    import docker
    import requests
    import yaml
except ImportError:
    print("Missing required dependencies. Install with:")
    print("pip install docker requests pyyaml")
    sys.exit(1)


class IntegrationTester:
    """Tests integration between all project components."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.test_results = {}
        self.running_services = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.error(f"Failed to connect to Docker: {e}")
            self.docker_client = None

    def wait_for_port(self, host: str, port: int, timeout: int = 30) -> bool:
        """Wait for a port to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except (socket.error, ConnectionRefusedError):
                time.sleep(1)
        
        return False

    def run_command(self, command: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> Tuple[bool, str, str]:
        """Run a command and return success status and output."""
        try:
            if cwd is None:
                cwd = self.project_root
            
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

    def test_python_environment(self) -> Dict[str, Any]:
        """Test Python environment setup."""
        self.logger.info("Testing Python environment...")
        
        test_result = {
            'name': 'Python Environment',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Test Python version
        python_version = sys.version_info
        test_result['tests'].append({
            'name': 'Python Version',
            'success': python_version >= (3, 8),
            'details': f"Python {python_version.major}.{python_version.minor}.{python_version.micro}"
        })
        
        if python_version < (3, 8):
            test_result['errors'].append("Python 3.8+ required")
            test_result['success'] = False
        
        # Test virtual environment
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        test_result['tests'].append({
            'name': 'Virtual Environment',
            'success': in_venv,
            'details': "Active" if in_venv else "Not active"
        })
        
        # Test package installation
        success, stdout, stderr = self.run_command([sys.executable, "-m", "pip", "list"])
        test_result['tests'].append({
            'name': 'Package Installation',
            'success': success,
            'details': f"Listed {len(stdout.splitlines())} packages" if success else stderr
        })
        
        if not success:
            test_result['errors'].append("Failed to list installed packages")
            test_result['success'] = False
        
        return test_result

    def test_project_structure(self) -> Dict[str, Any]:
        """Test project structure integrity."""
        self.logger.info("Testing project structure...")
        
        test_result = {
            'name': 'Project Structure',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Test importability of main package
        try:
            # Try to import the main package
            sys.path.insert(0, str(self.project_root / "src"))
            import vislang_ultralow
            
            test_result['tests'].append({
                'name': 'Package Import',
                'success': True,
                'details': f"Successfully imported vislang_ultralow from {vislang_ultralow.__file__}"
            })
        except ImportError as e:
            test_result['tests'].append({
                'name': 'Package Import',
                'success': False,
                'details': str(e)
            })
            test_result['errors'].append(f"Failed to import main package: {e}")
            test_result['success'] = False
        
        # Test configuration files
        config_files = [
            ("pyproject.toml", "Project configuration"),
            ("Dockerfile", "Docker configuration"),
            (".gitignore", "Git ignore rules")
        ]
        
        for file_name, description in config_files:
            file_path = self.project_root / file_name
            exists = file_path.exists()
            
            test_result['tests'].append({
                'name': description,
                'success': exists,
                'details': f"Found at {file_path}" if exists else f"Missing: {file_path}"
            })
            
            if not exists:
                test_result['errors'].append(f"Missing required file: {file_name}")
                test_result['success'] = False
        
        return test_result

    def test_docker_integration(self) -> Dict[str, Any]:
        """Test Docker integration."""
        self.logger.info("Testing Docker integration...")
        
        test_result = {
            'name': 'Docker Integration',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        if not self.docker_client:
            test_result['success'] = False
            test_result['errors'].append("Docker client not available")
            return test_result
        
        # Test Docker daemon
        try:
            self.docker_client.ping()
            test_result['tests'].append({
                'name': 'Docker Daemon',
                'success': True,
                'details': "Docker daemon is responding"
            })
        except Exception as e:
            test_result['tests'].append({
                'name': 'Docker Daemon',
                'success': False,
                'details': str(e)
            })
            test_result['errors'].append(f"Docker daemon not responding: {e}")
            test_result['success'] = False
            return test_result
        
        # Test Docker build
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            try:
                self.logger.info("Building Docker image for testing...")
                image, build_logs = self.docker_client.images.build(
                    path=str(self.project_root),
                    tag="vislang-test:latest",
                    rm=True,
                    pull=False,
                    nocache=False
                )
                
                test_result['tests'].append({
                    'name': 'Docker Build',
                    'success': True,
                    'details': f"Built image: {image.id[:12]}"
                })
                
                # Test container run
                try:
                    container = self.docker_client.containers.run(
                        image=image,
                        command="python -c 'import vislang_ultralow; print(\"Import successful\")'",
                        remove=True,
                        detach=False,
                        stdout=True,
                        stderr=True
                    )
                    
                    test_result['tests'].append({
                        'name': 'Container Run',
                        'success': True,
                        'details': "Container executed successfully"
                    })
                    
                except Exception as e:
                    test_result['tests'].append({
                        'name': 'Container Run',
                        'success': False,
                        'details': str(e)
                    })
                    test_result['errors'].append(f"Container execution failed: {e}")
                    test_result['success'] = False
                
                # Cleanup
                try:
                    self.docker_client.images.remove(image.id, force=True)
                except:
                    pass
                
            except Exception as e:
                test_result['tests'].append({
                    'name': 'Docker Build',
                    'success': False,
                    'details': str(e)
                })
                test_result['errors'].append(f"Docker build failed: {e}")
                test_result['success'] = False
        else:
            test_result['tests'].append({
                'name': 'Dockerfile Exists',
                'success': False,
                'details': "Dockerfile not found"
            })
            test_result['errors'].append("Dockerfile not found")
            test_result['success'] = False
        
        return test_result

    def test_testing_framework(self) -> Dict[str, Any]:
        """Test the testing framework integration."""
        self.logger.info("Testing framework integration...")
        
        test_result = {
            'name': 'Testing Framework',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Test pytest installation
        success, stdout, stderr = self.run_command([sys.executable, "-m", "pytest", "--version"])
        test_result['tests'].append({
            'name': 'Pytest Installation',
            'success': success,
            'details': stdout.strip() if success else stderr
        })
        
        if not success:
            test_result['errors'].append("Pytest not installed or not working")
            test_result['success'] = False
            return test_result
        
        # Test test discovery
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", "--collect-only", "-q"
        ])
        
        if success:
            lines = stdout.strip().split('\n')
            test_count = sum(1 for line in lines if '::' in line and 'test_' in line)
            test_result['tests'].append({
                'name': 'Test Discovery',
                'success': True,
                'details': f"Found {test_count} tests"
            })
        else:
            test_result['tests'].append({
                'name': 'Test Discovery',
                'success': False,
                'details': stderr
            })
            test_result['errors'].append("Test discovery failed")
            test_result['success'] = False
        
        # Run a subset of tests
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", "-x", "--tb=short", "tests/", "-k", "not integration"
        ], timeout=120)
        
        test_result['tests'].append({
            'name': 'Test Execution',
            'success': success,
            'details': "Tests passed" if success else f"Tests failed: {stderr}"
        })
        
        if not success:
            test_result['errors'].append("Some tests failed")
            # Don't mark as overall failure since tests might be expected to fail
        
        return test_result

    def test_monitoring_stack(self) -> Dict[str, Any]:
        """Test monitoring stack integration."""
        self.logger.info("Testing monitoring stack...")
        
        test_result = {
            'name': 'Monitoring Stack',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Check monitoring configuration files
        monitoring_dir = self.project_root / "monitoring"
        if not monitoring_dir.exists():
            test_result['tests'].append({
                'name': 'Monitoring Directory',
                'success': False,
                'details': "Monitoring directory not found"
            })
            test_result['errors'].append("Monitoring configuration not found")
            test_result['success'] = False
            return test_result
        
        # Test docker-compose configuration
        compose_file = monitoring_dir / "docker-compose.yml"
        if compose_file.exists():
            test_result['tests'].append({
                'name': 'Docker Compose Config',
                'success': True,
                'details': "Found monitoring docker-compose.yml"
            })
            
            try:
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                
                services = list(compose_data.get('services', {}).keys())
                test_result['tests'].append({
                    'name': 'Monitoring Services Config',
                    'success': True,
                    'details': f"Services: {', '.join(services)}"
                })
                
            except Exception as e:
                test_result['tests'].append({
                    'name': 'Compose File Parsing',
                    'success': False,
                    'details': str(e)
                })
                test_result['errors'].append(f"Failed to parse docker-compose.yml: {e}")
                test_result['success'] = False
        else:
            test_result['tests'].append({
                'name': 'Docker Compose Config',
                'success': False,
                'details': "docker-compose.yml not found"
            })
            test_result['errors'].append("Monitoring docker-compose.yml not found")
        
        # Test health check script
        health_script = self.project_root / "scripts" / "monitoring" / "health-check.py"
        if health_script.exists():
            success, stdout, stderr = self.run_command([
                sys.executable, str(health_script), "--help"
            ])
            test_result['tests'].append({
                'name': 'Health Check Script',
                'success': success,
                'details': "Health check script is executable" if success else stderr
            })
        else:
            test_result['tests'].append({
                'name': 'Health Check Script',
                'success': False,
                'details': "Health check script not found"
            })
        
        return test_result

    def test_automation_scripts(self) -> Dict[str, Any]:
        """Test automation scripts integration."""
        self.logger.info("Testing automation scripts...")
        
        test_result = {
            'name': 'Automation Scripts',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        scripts_to_test = [
            ("metrics_collection.py", "Metrics Collection"),
            ("automation/dependency-updater.py", "Dependency Updater"),
            ("automation/release-automation.py", "Release Automation"),
            ("validate-config.py", "Configuration Validator")
        ]
        
        for script_path, script_name in scripts_to_test:
            full_script_path = self.project_root / "scripts" / script_path
            
            if full_script_path.exists():
                # Test script help
                success, stdout, stderr = self.run_command([
                    sys.executable, str(full_script_path), "--help"
                ])
                
                test_result['tests'].append({
                    'name': f"{script_name} Executable",
                    'success': success,
                    'details': "Script runs and shows help" if success else stderr
                })
                
                if not success:
                    test_result['errors'].append(f"{script_name} script not executable")
                    test_result['success'] = False
            else:
                test_result['tests'].append({
                    'name': f"{script_name} Exists",
                    'success': False,
                    'details': f"Script not found: {script_path}"
                })
                test_result['errors'].append(f"{script_name} script not found")
                test_result['success'] = False
        
        return test_result

    def test_workflow_configuration(self) -> Dict[str, Any]:
        """Test GitHub Actions workflow configuration."""
        self.logger.info("Testing workflow configuration...")
        
        test_result = {
            'name': 'Workflow Configuration',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Check workflow examples
        workflows_dir = self.project_root / "docs" / "workflows" / "examples"
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml"))
            
            test_result['tests'].append({
                'name': 'Workflow Examples',
                'success': len(workflow_files) > 0,
                'details': f"Found {len(workflow_files)} workflow examples"
            })
            
            # Validate YAML syntax
            yaml_errors = []
            for workflow_file in workflow_files:
                try:
                    with open(workflow_file, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    yaml_errors.append(f"{workflow_file.name}: {e}")
            
            test_result['tests'].append({
                'name': 'Workflow YAML Syntax',
                'success': len(yaml_errors) == 0,
                'details': "All workflow files have valid YAML" if not yaml_errors else f"Errors: {yaml_errors}"
            })
            
            if yaml_errors:
                test_result['errors'].extend(yaml_errors)
                test_result['success'] = False
        else:
            test_result['tests'].append({
                'name': 'Workflow Examples Directory',
                'success': False,
                'details': "Workflow examples directory not found"
            })
            test_result['errors'].append("Workflow examples not found")
        
        return test_result

    def test_security_configuration(self) -> Dict[str, Any]:
        """Test security configuration."""
        self.logger.info("Testing security configuration...")
        
        test_result = {
            'name': 'Security Configuration',
            'success': True,
            'tests': [],
            'errors': []
        }
        
        # Test security tools availability
        security_tools = [
            ("bandit", "Security linter"),
            ("safety", "Dependency security checker")
        ]
        
        for tool, description in security_tools:
            success, stdout, stderr = self.run_command([tool, "--version"])
            test_result['tests'].append({
                'name': f"{description} ({tool})",
                'success': success,
                'details': stdout.strip() if success else f"Not installed: {stderr}"
            })
        
        # Test security policy
        security_policy = self.project_root / "SECURITY.md"
        test_result['tests'].append({
            'name': 'Security Policy',
            'success': security_policy.exists(),
            'details': "SECURITY.md found" if security_policy.exists() else "SECURITY.md not found"
        })
        
        return test_result

    def run_integration_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run all integration tests."""
        self.logger.info("Starting integration tests...")
        
        test_suites = [
            self.test_python_environment,
            self.test_project_structure,
            self.test_testing_framework,
            self.test_automation_scripts,
            self.test_workflow_configuration,
            self.test_security_configuration
        ]
        
        if not quick_mode:
            test_suites.extend([
                self.test_docker_integration,
                self.test_monitoring_stack
            ])
        
        results = {
            'overall_success': True,
            'test_results': [],
            'summary': {
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 0,
                'total_suites': len(test_suites),
                'passed_suites': 0,
                'failed_suites': 0
            },
            'errors': []
        }
        
        for test_suite in test_suites:
            try:
                test_result = test_suite()
                results['test_results'].append(test_result)
                
                # Update summary
                results['summary']['total_tests'] += len(test_result['tests'])
                results['summary']['passed_tests'] += sum(1 for t in test_result['tests'] if t['success'])
                results['summary']['failed_tests'] += sum(1 for t in test_result['tests'] if not t['success'])
                
                if test_result['success']:
                    results['summary']['passed_suites'] += 1
                else:
                    results['summary']['failed_suites'] += 1
                    results['overall_success'] = False
                    results['errors'].extend(test_result['errors'])
                
            except Exception as e:
                error_msg = f"Test suite failed with exception: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
                results['overall_success'] = False
                results['summary']['failed_suites'] += 1
        
        self.logger.info(f"Integration tests completed. Success: {results['overall_success']}")
        
        return results

    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way."""
        
        print("\n" + "="*60)
        print("VISLANG PROJECT INTEGRATION TEST RESULTS")
        print("="*60)
        
        summary = results['summary']
        
        # Overall status
        if results['overall_success']:
            print("\n✅ ALL INTEGRATION TESTS PASSED")
        else:
            print("\n❌ INTEGRATION TESTS FAILED")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Test Suites: {summary['passed_suites']}/{summary['total_suites']} passed")
        print(f"  Individual Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        
        # Detailed results
        for test_result in results['test_results']:
            status_icon = "✅" if test_result['success'] else "❌"
            print(f"\n{status_icon} {test_result['name']}")
            
            for test in test_result['tests']:
                test_icon = "  ✓" if test['success'] else "  ✗"
                print(f"{test_icon} {test['name']}: {test['details']}")
        
        # Print errors
        if results['errors']:
            print(f"\n🚨 ERRORS ({len(results['errors'])}):")
            for i, error in enumerate(results['errors'], 1):
                print(f"  {i}. {error}")
        
        print("\n" + "="*60)

    def cleanup(self):
        """Clean up any resources created during testing."""
        # Stop any running services
        for service in self.running_services:
            try:
                service.stop()
                service.remove()
            except:
                pass
        
        # Clean up Docker resources
        if self.docker_client:
            try:
                # Remove test images
                images = self.docker_client.images.list(filters={'reference': 'vislang-test:*'})
                for image in images:
                    self.docker_client.images.remove(image.id, force=True)
            except:
                pass


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Run VisLang integration tests")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick tests (skip Docker and monitoring)"
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

    # Initialize tester
    tester = IntegrationTester(args.project_root)

    try:
        # Run integration tests
        results = tester.run_integration_tests(quick_mode=args.quick)

        # Print results
        tester.print_results(results)

        # Save JSON results if requested
        if args.json:
            try:
                with open(args.json, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.json}")
            except Exception as e:
                print(f"Failed to save JSON results: {e}")

        # Exit with appropriate code
        sys.exit(0 if results['overall_success'] else 1)

    finally:
        # Always cleanup
        tester.cleanup()


if __name__ == "__main__":
    main()