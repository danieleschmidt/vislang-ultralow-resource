# Automation Scripts

This directory contains automation scripts for the VisLang-UltraLow-Resource project.

## Available Scripts

### 📊 Metrics Collection (`metrics_collection.py`)

Collects various project metrics from different sources and updates the project metrics file.

```bash
# Collect all metrics
python scripts/metrics_collection.py

# Collect specific metric types
python scripts/metrics_collection.py --types repository quality security

# Output summary to file
python scripts/metrics_collection.py --output-summary metrics-summary.md

# Verbose output
python scripts/metrics_collection.py --verbose
```

**Features:**
- GitHub repository statistics (stars, forks, commits, PRs, issues)
- Code quality metrics (test coverage, documentation ratio)
- Security vulnerability scanning
- Infrastructure metrics (CPU, memory, disk usage)
- Build and CI/CD metrics
- Health score calculation

**Required Environment Variables:**
- `GITHUB_TOKEN`: For GitHub API access
- `GITHUB_REPOSITORY_OWNER`: Repository owner (default: danieleschmidt)
- `GITHUB_REPOSITORY_NAME`: Repository name (default: vislang-ultralow-resource)

### 🔄 Dependency Updates (`dependency_update.py`)

Automated dependency management with security vulnerability checking.

```bash
# Check for outdated packages
python scripts/dependency_update.py --action check

# Update security vulnerabilities
python scripts/dependency_update.py --action security

# Update to latest minor versions
python scripts/dependency_update.py --action update --update-type minor

# Generate dependency report
python scripts/dependency_update.py --action report --output-report dep-report.md

# Auto-merge security updates
python scripts/dependency_update.py --action security --auto-merge
```

**Features:**
- Security vulnerability detection (pip-audit, safety)
- Categorized updates (patch, minor, major)
- Automated pull request creation
- Integration with GitHub API
- Backup and rollback capabilities

**Required Dependencies:**
```bash
pip install requests packaging
pip install safety pip-audit  # Optional for security scanning
```

### 🧹 Repository Maintenance (`repository_maintenance.py`)

Comprehensive repository maintenance and health checking.

```bash
# Generate maintenance report
python scripts/repository_maintenance.py --tasks generate-report

# Clean up old branches (dry run)
python scripts/repository_maintenance.py --tasks cleanup-branches --dry-run

# Update README badges
python scripts/repository_maintenance.py --tasks update-badges

# Check for broken links
python scripts/repository_maintenance.py --tasks check-links

# Validate project structure
python scripts/repository_maintenance.py --tasks validate-structure

# Run all maintenance tasks
python scripts/repository_maintenance.py --tasks cleanup-branches update-badges check-links validate-structure generate-report

# Save report to file
python scripts/repository_maintenance.py --output-report maintenance-report.md
```

**Features:**
- Old branch cleanup (merged branches older than 30 days)
- README badge updates (Python version, license, tests)
- Broken link detection in documentation
- Project structure validation
- Git statistics and contributor analysis
- Health score calculation

**Required Dependencies:**
```bash
pip install requests
```

### 🧪 Test Runner (`run_tests.sh`)

Comprehensive test execution script with coverage reporting.

```bash
# Run all tests
./scripts/run_tests.sh

# Run only unit tests
./scripts/run_tests.sh unit

# Run with coverage
./scripts/run_tests.sh --coverage

# Run with verbose output
./scripts/run_tests.sh --verbose

# Generate HTML coverage report
./scripts/run_tests.sh --coverage --html
```

## Usage in CI/CD

### GitHub Actions Integration

```yaml
# In .github/workflows/metrics.yml
- name: Collect metrics
  run: |
    python scripts/metrics_collection.py --output-summary metrics-summary.md
    
- name: Update dependencies
  run: |
    python scripts/dependency_update.py --action security --auto-merge

- name: Repository maintenance
  run: |
    python scripts/repository_maintenance.py --tasks cleanup-branches update-badges
```

### Scheduled Tasks

```yaml
# Weekly maintenance
schedule:
  - cron: '0 2 * * 1'  # Monday 2 AM
jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run maintenance
      run: |
        python scripts/repository_maintenance.py --tasks cleanup-branches
        python scripts/dependency_update.py --action security
```

## Configuration

### Environment Variables

```bash
# GitHub integration
export GITHUB_TOKEN="your_github_token"
export GITHUB_REPOSITORY_OWNER="danieleschmidt"
export GITHUB_REPOSITORY_NAME="vislang-ultralow-resource"

# Optional configurations
export METRICS_CONFIG_PATH=".github/project-metrics.json"
export REQUIREMENTS_FILE="requirements.txt"
```

### Project Metrics Configuration

The metrics are stored in `.github/project-metrics.json` and include:

- **Repository Metrics**: Health score, activity, quality, performance
- **Application Metrics**: Processing, ML metrics, data quality, API metrics
- **Business Metrics**: Impact, usage, growth
- **Infrastructure Metrics**: Availability, performance, costs
- **Security Metrics**: Vulnerability management, compliance, incidents
- **Quality Metrics**: Code quality, documentation, testing
- **Team Metrics**: Productivity, collaboration, satisfaction

## Best Practices

### 1. Regular Execution

- Run metrics collection daily
- Check dependencies weekly
- Perform maintenance monthly
- Generate reports for stakeholders

### 2. Integration with Monitoring

```python
# Send metrics to monitoring system
import json
from metrics_collection import MetricsCollector

collector = MetricsCollector()
collector.run_collection()

# Post to monitoring endpoint
with open('.github/project-metrics.json', 'r') as f:
    metrics = json.load(f)
    # send_to_monitoring_system(metrics)
```

### 3. Alerting Setup

```bash
# Check for critical issues
python scripts/metrics_collection.py --types security
if [ $? -ne 0 ]; then
    echo "Critical security issues found!"
    # Send alert
fi
```

### 4. Documentation Updates

```bash
# Auto-update documentation
python scripts/repository_maintenance.py --tasks update-badges
git add README.md
git commit -m "docs: update badges automatically"
```

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   ```bash
   chmod +x scripts/*.py
   ```

2. **Missing Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **GitHub API Rate Limits**:
   - Use personal access token
   - Implement caching
   - Reduce API calls frequency

4. **Git Authentication**:
   ```bash
   git config --global user.email "action@github.com"
   git config --global user.name "GitHub Action"
   ```

### Debugging

Enable verbose logging:
```bash
python scripts/metrics_collection.py --verbose
python scripts/dependency_update.py --verbose
python scripts/repository_maintenance.py --verbose
```

Check logs:
```bash
tail -f logs/automation.log
```

## Security Considerations

1. **Token Management**: Store GitHub tokens in secrets, not in code
2. **Branch Protection**: Scripts respect branch protection rules
3. **Pull Request Reviews**: Security updates can auto-merge, others require review
4. **Audit Trail**: All actions are logged and tracked
5. **Backup Strategy**: Scripts create backups before making changes

## Contributing

When adding new automation scripts:

1. Follow the existing naming convention
2. Include comprehensive help text and examples
3. Add error handling and logging
4. Write unit tests
5. Update this README
6. Add to CI/CD workflows if appropriate

## Support

For issues with automation scripts:

1. Check the logs for detailed error messages
2. Verify required dependencies are installed
3. Ensure environment variables are set correctly
4. Test scripts locally before running in CI/CD
5. Create an issue with the `automation` label