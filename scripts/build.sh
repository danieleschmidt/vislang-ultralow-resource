#!/bin/bash
# Build script for VisLang-UltraLow-Resource
# Supports multi-architecture builds and different targets

set -euo pipefail

# Configuration
REGISTRY=${REGISTRY:-"ghcr.io/danieleschmidt"}
IMAGE_NAME=${IMAGE_NAME:-"vislang-ultralow-resource"}
VERSION=${VERSION:-$(cat pyproject.toml | grep -E '^version = ' | cut -d'"' -f2)}
PLATFORMS=${PLATFORMS:-"linux/amd64,linux/arm64"}
PUSH=${PUSH:-false}
TARGET=${TARGET:-"production"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[BUILD]${NC} $1"
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

# Check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running or not accessible"
    fi
}

# Check if buildx is available for multi-arch builds
check_buildx() {
    if ! docker buildx version >/dev/null 2>&1; then
        error "Docker buildx is not available. Please install Docker with buildx support."
    fi
}

# Setup buildx builder for multi-architecture builds
setup_builder() {
    local builder_name="vislang-builder"
    
    if ! docker buildx inspect $builder_name >/dev/null 2>&1; then
        log "Creating buildx builder for multi-architecture builds..."
        docker buildx create --name $builder_name --use --bootstrap
    else
        log "Using existing buildx builder: $builder_name"
        docker buildx use $builder_name
    fi
}

# Build function
build_image() {
    local tag="$1"
    local target="$2"
    local platforms="$3"
    local push_flag="$4"
    
    log "Building image: $tag"
    log "Target: $target"
    log "Platforms: $platforms"
    
    local build_args=""
    build_args+=" --platform $platforms"
    build_args+=" --target $target"
    build_args+=" --tag $tag"
    build_args+=" --file Dockerfile"
    
    # Add build metadata
    build_args+=" --label org.opencontainers.image.title=VisLang-UltraLow-Resource"
    build_args+=" --label org.opencontainers.image.description='Dataset builder and training framework for visual-language models in ultra-low-resource languages'"
    build_args+=" --label org.opencontainers.image.version=$VERSION"
    build_args+=" --label org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    build_args+=" --label org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    build_args+=" --label org.opencontainers.image.source=https://github.com/danieleschmidt/vislang-ultralow-resource"
    build_args+=" --label org.opencontainers.image.licenses=MIT"
    
    if [ "$push_flag" = true ]; then
        build_args+=" --push"
        log "Image will be pushed to registry"
    else
        build_args+=" --load"
        log "Image will be loaded locally (single platform only)"
    fi
    
    # Execute build
    docker buildx build $build_args .
}

# Health check function
health_check() {
    local image="$1"
    log "Running health check for image: $image"
    
    # Start container for health check
    local container_id=$(docker run -d --name vislang-health-check -p 8000:8000 $image)
    
    # Wait for container to start
    sleep 10
    
    # Check health endpoint
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        success "Health check passed"
    else
        error "Health check failed"
    fi
    
    # Cleanup
    docker stop $container_id >/dev/null 2>&1 || true
    docker rm $container_id >/dev/null 2>&1 || true
}

# Security scan function
security_scan() {
    local image="$1"
    log "Running security scan for image: $image"
    
    if command -v trivy >/dev/null 2>&1; then
        trivy image --severity HIGH,CRITICAL $image
    elif command -v docker-scan >/dev/null 2>&1; then
        docker scan $image
    else
        warn "No security scanner available. Install trivy or docker-scan for security scanning."
    fi
}

# Size analysis function
analyze_size() {
    local image="$1"
    log "Analyzing image size: $image"
    
    docker images $image --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    if command -v dive >/dev/null 2>&1; then
        log "Running detailed layer analysis with dive..."
        dive $image
    fi
}

# Main build process
main() {
    log "Starting build process for VisLang-UltraLow-Resource"
    log "Version: $VERSION"
    log "Target: $TARGET"
    log "Platforms: $PLATFORMS"
    
    # Pre-flight checks
    check_docker
    check_buildx
    
    # Setup multi-arch builder if building for multiple platforms
    if [[ "$PLATFORMS" == *","* ]] || [ "$PUSH" = true ]; then
        setup_builder
    fi
    
    # Build tags
    local base_tag="$REGISTRY/$IMAGE_NAME"
    local version_tag="$base_tag:$VERSION"
    local latest_tag="$base_tag:latest"
    local target_tag="$base_tag:$TARGET"
    
    # Build main image
    build_image "$version_tag" "$TARGET" "$PLATFORMS" "$PUSH"
    
    # Tag as latest if this is a production build
    if [ "$TARGET" = "production" ] && [ "$PUSH" = true ]; then
        log "Tagging as latest..."
        docker buildx build \
            --platform "$PLATFORMS" \
            --target "$TARGET" \
            --tag "$latest_tag" \
            --push \
            .
    fi
    
    # Build development image if requested
    if [ "$TARGET" = "development" ] || [ "$TARGET" = "all" ]; then
        log "Building development image..."
        build_image "$base_tag:dev" "development" "$PLATFORMS" "$PUSH"
    fi
    
    # Build minimal image if requested
    if [ "$TARGET" = "minimal" ] || [ "$TARGET" = "all" ]; then
        log "Building minimal image..."
        build_image "$base_tag:minimal" "minimal" "$PLATFORMS" "$PUSH"
    fi
    
    # Run checks on single-platform builds only
    if [[ "$PLATFORMS" != *","* ]] && [ "$PUSH" = false ]; then
        health_check "$version_tag"
        security_scan "$version_tag"
        analyze_size "$version_tag"
    fi
    
    success "Build completed successfully!"
    log "Built image: $version_tag"
    
    if [ "$PUSH" = true ]; then
        log "Image pushed to: $version_tag"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-push)
            PUSH=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --registry REGISTRY     Container registry (default: ghcr.io/danieleschmidt)"
            echo "  --image-name NAME       Image name (default: vislang-ultralow-resource)"
            echo "  --version VERSION       Image version (default: from pyproject.toml)"
            echo "  --platforms PLATFORMS   Target platforms (default: linux/amd64,linux/arm64)"
            echo "  --target TARGET         Build target (default: production)"
            echo "  --push                  Push to registry"
            echo "  --no-push               Don't push to registry (default)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Targets:"
            echo "  development             Development image with all tools"
            echo "  production              Production image (default)"
            echo "  minimal                 Minimal production image"
            echo "  all                     Build all targets"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build production image locally"
            echo "  $0 --target development               # Build development image"
            echo "  $0 --push --target production         # Build and push production image"
            echo "  $0 --platforms linux/amd64 --no-push # Build for single platform"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main