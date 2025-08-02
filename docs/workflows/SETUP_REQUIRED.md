# Manual GitHub Workflow Setup Required

Due to GitHub App permission limitations, the following workflows cannot be automatically created and must be manually implemented by repository maintainers.

## Required Actions

### 1. Create Workflow Files

Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
cp docs/workflows/examples/dependencies.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add the following secrets in repository settings:

#### Required Secrets
- `PYPI_API_TOKEN`: For publishing to PyPI
- `CODECOV_TOKEN`: For code coverage reporting
- `SNYK_TOKEN`: For security vulnerability scanning

#### Optional Secrets
- `SLACK_WEBHOOK`: For team notifications
- `GITLEAKS_LICENSE`: For enhanced secret detection

### 3. Enable Security Features

In repository settings, enable:
- [ ] Code scanning (CodeQL)
- [ ] Secret scanning
- [ ] Dependency scanning
- [ ] Security advisories

### 4. Configure Branch Protection Rules

For the `main` branch, enable:
- [ ] Require pull request reviews
- [ ] Require status checks to pass before merging
- [ ] Required status checks:
  - `Code Quality Checks`
  - `Security Analysis`
  - `Test Suite`
  - `Build and Test Package`
- [ ] Require branches to be up to date before merging
- [ ] Restrict pushes that create files
- [ ] Require signed commits (recommended)

### 5. Set up Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
    open-pull-requests-limit: 5
    commit-message:
      prefix: "deps"
      include: "scope"
    labels:
      - "dependencies"
      - "automated"
    reviewers:
      - "team-leads"
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "github-actions"
      - "dependencies"
```

### 6. Configure Issue Templates

Create `.github/ISSUE_TEMPLATE/`:

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

#### Bug Report Template (`.github/ISSUE_TEMPLATE/bug_report.yml`)

```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "needs-triage"]
assignees: []
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: contact
    attributes:
      label: Contact Details
      description: How can we get in touch with you if we need more info?
      placeholder: ex. email@example.com
    validations:
      required: false
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: How can we reproduce this issue?
      placeholder: |
        1. Run command '...'
        2. See error
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant log output
      description: Please copy and paste any relevant log output. This will be automatically formatted into code, so no need for backticks.
      render: shell
  - type: dropdown
    id: version
    attributes:
      label: Version
      description: What version of VisLang are you running?
      options:
        - latest
        - 1.0.0
        - Other (please specify in description)
    validations:
      required: true
```

#### Feature Request Template (`.github/ISSUE_TEMPLATE/feature_request.yml`)

```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[Feature]: "
labels: ["enhancement", "needs-triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  - type: textarea
    id: problem
    attributes:
      label: Is your feature request related to a problem?
      description: A clear description of what the problem is.
      placeholder: I'm always frustrated when...
    validations:
      required: false
  - type: textarea
    id: solution
    attributes:
      label: Describe the solution you'd like
      description: A clear description of what you want to happen.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Describe alternatives you've considered
      description: A clear description of any alternative solutions or features you've considered.
    validations:
      required: false
  - type: textarea
    id: context
    attributes:
      label: Additional context
      description: Add any other context or screenshots about the feature request here.
    validations:
      required: false
```

### 7. Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

- [ ] Tests pass locally with my changes
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## Documentation

- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added necessary comments to complex code

## Security

- [ ] I have reviewed my code for potential security issues
- [ ] No sensitive information (keys, passwords, tokens) is committed
- [ ] Dependencies are from trusted sources

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes do not generate any new linting errors
- [ ] I have checked my code and corrected any misspellings

## Related Issues

Fixes #(issue number)

## Additional Notes

Any additional information that reviewers should know.
```

### 8. CodeQL Configuration

Create `.github/codeql/codeql-config.yml`:

```yaml
name: "VisLang CodeQL Config"

disable-default-queries: false

queries:
  - uses: security-and-quality
  - uses: security-extended

paths-ignore:
  - "tests/"
  - "docs/"
  - "scripts/"

paths:
  - "src/"

query-filters:
  - exclude:
      id: py/unused-import
  - exclude:
      id: py/similar-function
```

### 9. Repository Settings Configuration

#### General Settings
- [ ] Allow squash merging
- [ ] Allow merge commits: ❌ Disable
- [ ] Allow rebase merging: ❌ Disable
- [ ] Automatically delete head branches: ✅ Enable

#### Security & Analysis
- [ ] Private vulnerability reporting: ✅ Enable
- [ ] Dependency graph: ✅ Enable
- [ ] Dependabot alerts: ✅ Enable
- [ ] Dependabot security updates: ✅ Enable
- [ ] Secret scanning: ✅ Enable
- [ ] Push protection: ✅ Enable

#### Actions Permissions
- [ ] Allow all actions and reusable workflows
- [ ] Allow actions created by GitHub: ✅ Enable
- [ ] Allow actions by Marketplace verified creators: ✅ Enable

### 10. Environment Configuration

Create the following environments in repository settings:

#### `release` Environment
- Protection rules:
  - Required reviewers: Repository maintainers
  - Wait timer: 5 minutes
- Environment secrets:
  - `PYPI_API_TOKEN`

#### `pypi` Environment  
- Protection rules:
  - Required reviewers: Repository maintainers
- Environment secrets:
  - `PYPI_API_TOKEN`

### 11. Webhooks and Integrations

Consider setting up:
- [ ] Slack integration for notifications
- [ ] Code quality services (SonarCloud, Codacy)
- [ ] Security scanning services (Snyk, WhiteSource)

## Verification Steps

After implementing the workflows:

1. **Test CI Pipeline**:
   ```bash
   # Create a test branch and push changes
   git checkout -b test/ci-setup
   echo "# Test" >> test-file.md
   git add test-file.md
   git commit -m "test: verify CI pipeline"
   git push -u origin test/ci-setup
   ```

2. **Create Test PR**: Open a pull request to verify all checks run

3. **Test Security Scanning**: Push a commit with a test secret (then remove it)

4. **Test Release Process**: Create a test tag to verify release workflow

5. **Verify Dependabot**: Check that Dependabot creates update PRs

## Troubleshooting

### Common Issues

1. **Workflow Permission Errors**:
   - Verify repository token permissions
   - Check if organization restricts workflow permissions

2. **Secret Not Found Errors**:
   - Verify secrets are added to correct environment
   - Check secret names match workflow references

3. **Status Check Failures**:
   - Ensure branch protection required checks match workflow job names
   - Verify workflow files are in correct location

4. **Auto-merge Not Working**:
   - Check branch protection settings
   - Verify status checks are configured correctly

### Support

If you encounter issues during setup:
1. Check the workflow logs for detailed error messages
2. Review GitHub's documentation on Actions and security features
3. Create an issue in this repository with the `setup-help` label

## Implementation Checklist

- [ ] Copy workflow files from examples
- [ ] Configure repository secrets
- [ ] Enable security features
- [ ] Set up branch protection rules
- [ ] Create Dependabot configuration
- [ ] Add issue and PR templates
- [ ] Configure CodeQL
- [ ] Set repository settings
- [ ] Create release and PyPI environments
- [ ] Test all workflows
- [ ] Document any customizations

---

**Note**: This setup is required due to GitHub App permission limitations. Once implemented, the workflows will provide comprehensive CI/CD, security scanning, and automated dependency management for the VisLang project.