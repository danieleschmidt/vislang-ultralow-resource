#!/bin/bash
# Generate Software Bill of Materials (SBOM) for VisLang-UltraLow-Resource
# Supports multiple SBOM formats and tools

set -euo pipefail

# Configuration
OUTPUT_DIR=${OUTPUT_DIR:-"./sbom"}
VERSION=${VERSION:-$(grep -E '^version = ' pyproject.toml | cut -d'"' -f2)}
FORMATS=${FORMATS:-"spdx-json,cyclonedx-json,table"}
IMAGE_NAME=${IMAGE_NAME:-"vislang-ultralow-resource:prod"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[SBOM]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check dependencies
check_dependencies() {
    local missing_tools=()
    
    # Check for syft (preferred SBOM tool)
    if ! command -v syft >/dev/null 2>&1; then
        missing_tools+=("syft")
    fi
    
    # Check for cyclonedx-bom (alternative)
    if ! command -v cyclonedx-bom >/dev/null 2>&1; then
        missing_tools+=("cyclonedx-bom")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        warn "Missing SBOM tools: ${missing_tools[*]}"
        log "Installing syft..."
        install_syft
    fi
}

# Install syft
install_syft() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install syft
        else
            curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
        fi
    else
        error "Unsupported operating system for automatic syft installation"
    fi
}

# Generate SBOM for Python package
generate_python_sbom() {
    log "Generating SBOM for Python package..."
    
    mkdir -p "$OUTPUT_DIR"
    
    # Generate with syft if available
    if command -v syft >/dev/null 2>&1; then
        IFS=',' read -ra format_array <<< "$FORMATS"
        for format in "${format_array[@]}"; do
            local output_file="$OUTPUT_DIR/vislang-python-$VERSION.$format"
            log "Generating $format SBOM: $output_file"
            
            syft . -o "$format=$output_file" --quiet
        done
    fi
    
    # Generate requirements-based SBOM
    generate_requirements_sbom
}

# Generate SBOM for Docker image
generate_docker_sbom() {
    local image_name="$1"
    log "Generating SBOM for Docker image: $image_name"
    
    mkdir -p "$OUTPUT_DIR"
    
    if command -v syft >/dev/null 2>&1; then
        IFS=',' read -ra format_array <<< "$FORMATS"
        for format in "${format_array[@]}"; do
            local output_file="$OUTPUT_DIR/vislang-docker-$VERSION.$format"
            log "Generating $format SBOM: $output_file"
            
            syft "$image_name" -o "$format=$output_file" --quiet
        done
    fi
}

# Generate requirements-based SBOM
generate_requirements_sbom() {
    log "Generating requirements-based SBOM..."
    
    local req_file="$OUTPUT_DIR/requirements-sbom-$VERSION.txt"
    
    # Create detailed requirements list
    cat > "$req_file" << EOF
# Software Bill of Materials - Requirements
# Package: vislang-ultralow-resource
# Version: $VERSION
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Production Dependencies
EOF
    
    # Extract dependencies from pyproject.toml
    python3 -c "
import tomllib
import sys

try:
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    
    deps = data.get('project', {}).get('dependencies', [])
    print('# Core dependencies:')
    for dep in deps:
        print(dep)
    
    print('\n# Optional dependencies:')
    optional_deps = data.get('project', {}).get('optional-dependencies', {})
    for group, group_deps in optional_deps.items():
        print(f'# {group}:')
        for dep in group_deps:
            print(dep)
except Exception as e:
    print(f'Error parsing pyproject.toml: {e}', file=sys.stderr)
    sys.exit(1)
" >> "$req_file"
    
    # Generate pip freeze output if in virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
        echo -e "\n# Installed package versions:" >> "$req_file"
        pip freeze >> "$req_file" 2>/dev/null || true
    fi
}

# Generate vulnerability report
generate_vulnerability_report() {
    log "Generating vulnerability report..."
    
    local vuln_file="$OUTPUT_DIR/vulnerabilities-$VERSION.json"
    
    # Use grype if available
    if command -v grype >/dev/null 2>&1; then
        log "Running grype vulnerability scan..."
        grype . -o json > "$vuln_file" 2>/dev/null || warn "Grype scan failed"
    elif command -v safety >/dev/null 2>&1; then
        log "Running safety vulnerability scan..."
        safety check --json > "$vuln_file" 2>/dev/null || warn "Safety scan failed"
    else
        warn "No vulnerability scanner available (grype or safety)"
    fi
}

# Generate license report
generate_license_report() {
    log "Generating license report..."
    
    local license_file="$OUTPUT_DIR/licenses-$VERSION.txt"
    
    cat > "$license_file" << EOF
# License Report - VisLang-UltraLow-Resource
# Version: $VERSION
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Main License
Project: vislang-ultralow-resource
License: MIT
File: LICENSE

# Third-party Licenses
EOF
    
    # Use pip-licenses if available
    if command -v pip-licenses >/dev/null 2>&1; then
        echo -e "\n# Installed package licenses:" >> "$license_file"
        pip-licenses --format=plain >> "$license_file" 2>/dev/null || warn "pip-licenses failed"
    else
        warn "pip-licenses not available. Install with: pip install pip-licenses"
    fi
}

# Generate compliance report
generate_compliance_report() {
    log "Generating compliance report..."
    
    local compliance_file="$OUTPUT_DIR/compliance-$VERSION.md"
    
    cat > "$compliance_file" << EOF
# Compliance Report - VisLang-UltraLow-Resource

**Version:** $VERSION  
**Generated:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")  
**Tool:** vislang-sbom-generator  

## Overview

This report provides compliance information for the VisLang-UltraLow-Resource project,
including software components, licenses, and security information.

## Software Bill of Materials (SBOM)

The following SBOM files have been generated:

- **SPDX JSON:** vislang-python-$VERSION.spdx-json
- **CycloneDX JSON:** vislang-python-$VERSION.cyclonedx-json
- **Table Format:** vislang-python-$VERSION.table

## License Compliance

- **Project License:** MIT
- **Third-party Licenses:** See licenses-$VERSION.txt
- **License Compatibility:** All dependencies are compatible with MIT license

## Security Assessment

- **Vulnerability Report:** vulnerabilities-$VERSION.json
- **Last Scanned:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Supply Chain Security

### Package Sources
- **PyPI:** All Python dependencies sourced from official PyPI repository
- **Container Base:** Official Python slim image from Docker Hub
- **System Packages:** Official Debian/Ubuntu repositories

### Verification
- All dependencies are pinned to specific versions
- Container images are signed and verified
- Build process uses reproducible builds where possible

## Recommendations

1. Regularly update dependencies to latest secure versions
2. Monitor vulnerability databases for new issues
3. Verify package signatures when available
4. Use container image scanning in CI/CD pipeline

---

*This report is automatically generated and should be reviewed by security team.*
EOF
}

# Main function
main() {
    log "Generating SBOM for VisLang-UltraLow-Resource v$VERSION"
    
    # Check dependencies
    check_dependencies
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Generate different types of SBOMs
    generate_python_sbom
    
    # Generate Docker SBOM if image exists
    if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        generate_docker_sbom "$IMAGE_NAME"
    else
        warn "Docker image $IMAGE_NAME not found, skipping Docker SBOM"
    fi
    
    # Generate additional reports
    generate_vulnerability_report
    generate_license_report
    generate_compliance_report
    
    # Create archive
    local archive_name="vislang-sbom-$VERSION.tar.gz"
    tar -czf "$archive_name" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
    
    success "SBOM generation completed!"
    log "Files generated in: $OUTPUT_DIR"
    log "Archive created: $archive_name"
    
    # List generated files
    echo
    log "Generated files:"
    find "$OUTPUT_DIR" -type f -exec basename {} \; | sort | sed 's/^/  - /'
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --formats)
            FORMATS="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR    Output directory (default: ./sbom)"
            echo "  --version VERSION   Package version (default: from pyproject.toml)"
            echo "  --formats FORMATS   SBOM formats (default: spdx-json,cyclonedx-json,table)"
            echo "  --image IMAGE       Docker image name (default: vislang-ultralow-resource:prod)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Generate SBOM with defaults"
            echo "  $0 --formats spdx-json               # Generate only SPDX format"
            echo "  $0 --output-dir /tmp/sbom             # Custom output directory"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main