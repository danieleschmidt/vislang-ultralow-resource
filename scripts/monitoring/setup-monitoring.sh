#!/bin/bash
# Setup monitoring stack for VisLang-UltraLow-Resource
# Configures Prometheus, Grafana, Alertmanager, and related monitoring tools

set -euo pipefail

# Configuration
MONITORING_DIR="${MONITORING_DIR:-./monitoring}"
DATA_DIR="${DATA_DIR:-./data}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[MONITORING]${NC} $1"
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
    
    if ! command -v docker >/dev/null 2>&1; then
        missing_tools+=("docker")
    fi
    
    if ! command -v docker-compose >/dev/null 2>&1; then
        missing_tools+=("docker-compose")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
    fi
}

# Create monitoring directories
setup_directories() {
    log "Setting up monitoring directories..."
    
    local dirs=(
        "$DATA_DIR/prometheus"
        "$DATA_DIR/grafana"
        "$DATA_DIR/alertmanager"
        "$DATA_DIR/loki"
        "$MONITORING_DIR/grafana/provisioning/dashboards"
        "$MONITORING_DIR/grafana/provisioning/datasources"
        "$MONITORING_DIR/grafana/provisioning/alerting"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
    
    # Set permissions for Grafana
    chmod -R 777 "$DATA_DIR/grafana" 2>/dev/null || warn "Could not set Grafana permissions"
}

# Configure Grafana datasources
setup_grafana_datasources() {
    log "Setting up Grafana datasources..."
    
    cat > "$MONITORING_DIR/grafana/provisioning/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true
    
  - name: AlertManager
    type: alertmanager
    access: proxy
    url: http://alertmanager:9093
    editable: true
EOF
}

# Configure Grafana dashboards provisioning
setup_grafana_dashboards() {
    log "Setting up Grafana dashboard provisioning..."
    
    cat > "$MONITORING_DIR/grafana/provisioning/dashboards/dashboards.yml" << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
}

# Configure alertmanager
setup_alertmanager() {
    log "Setting up Alertmanager configuration..."
    
    local alertmanager_config="$MONITORING_DIR/alertmanager.yml"
    
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        log "Configuring Slack notifications..."
        sed -i "s|YOUR_SLACK_WEBHOOK_URL|$SLACK_WEBHOOK_URL|g" "$alertmanager_config"
    fi
    
    if [[ -n "$ALERT_WEBHOOK_URL" ]]; then
        log "Configuring webhook notifications..."
        sed -i "s|http://127.0.0.1:5001/|$ALERT_WEBHOOK_URL|g" "$alertmanager_config"
    fi
}

# Setup log rotation for monitoring components
setup_log_rotation() {
    log "Setting up log rotation..."
    
    cat > /tmp/monitoring-logrotate << EOF
/var/log/monitoring/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
    notifempty
    postrotate
        docker-compose -f docker-compose.yml restart prometheus grafana || true
    endscript
}
EOF
    
    if [[ -w /etc/logrotate.d/ ]]; then
        sudo mv /tmp/monitoring-logrotate /etc/logrotate.d/monitoring
        success "Log rotation configured"
    else
        warn "Could not configure log rotation (no write access to /etc/logrotate.d/)"
        rm -f /tmp/monitoring-logrotate
    fi
}

# Configure Prometheus targets
configure_prometheus_targets() {
    log "Configuring Prometheus targets..."
    
    # Check if custom targets file exists
    local targets_file="$MONITORING_DIR/targets.json"
    
    if [[ ! -f "$targets_file" ]]; then
        cat > "$targets_file" << EOF
[
  {
    "targets": ["vislang-app:9091"],
    "labels": {
      "job": "vislang-app",
      "service": "application"
    }
  },
  {
    "targets": ["postgres-exporter:9187"],
    "labels": {
      "job": "postgres",
      "service": "database"
    }
  },
  {
    "targets": ["redis-exporter:9121"],
    "labels": {
      "job": "redis",
      "service": "cache"
    }
  }
]
EOF
        log "Created default targets configuration"
    fi
}

# Install monitoring exporters
install_exporters() {
    log "Setting up monitoring exporters..."
    
    # Create docker-compose override for exporters
    cat > docker-compose.monitoring.yml << EOF
version: '3.8'

services:
  # PostgreSQL Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: vislang-postgres-exporter
    environment:
      DATA_SOURCE_NAME: "postgresql://vislang:vislang_password@postgres:5432/vislang_db?sslmode=disable"
    ports:
      - "9187:9187"
    depends_on:
      - postgres
    networks:
      - vislang-network
    restart: unless-stopped

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: vislang-redis-exporter
    environment:
      REDIS_ADDR: "redis://redis:6379"
    ports:
      - "9121:9121"
    depends_on:
      - redis
    networks:
      - vislang-network
    restart: unless-stopped

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    container_name: vislang-alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--web.route-prefix=/'
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager-data:/alertmanager
    networks:
      - vislang-network
    restart: unless-stopped

  # cAdvisor for container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: vislang-cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - vislang-network
    restart: unless-stopped

volumes:
  alertmanager-data:

networks:
  vislang-network:
    external: true
EOF
    
    log "Created monitoring services docker-compose file"
}

# Validate monitoring configuration
validate_configuration() {
    log "Validating monitoring configuration..."
    
    local errors=0
    
    # Check Prometheus config
    if command -v promtool >/dev/null 2>&1; then
        if ! promtool check config "$MONITORING_DIR/prometheus.yml"; then
            error "Prometheus configuration is invalid"
            ((errors++))
        fi
        
        if ! promtool check rules "$MONITORING_DIR/alert_rules.yml"; then
            error "Prometheus alert rules are invalid"
            ((errors++))
        fi
    else
        warn "promtool not available, skipping Prometheus config validation"
    fi
    
    # Check required files
    local required_files=(
        "$MONITORING_DIR/prometheus.yml"
        "$MONITORING_DIR/alert_rules.yml"
        "$MONITORING_DIR/alertmanager.yml"
        "$MONITORING_DIR/grafana/provisioning/datasources/prometheus.yml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            error "Required file missing: $file"
            ((errors++))
        fi
    done
    
    if [[ $errors -gt 0 ]]; then
        error "Configuration validation failed with $errors error(s)"
    fi
    
    success "Configuration validation passed"
}

# Start monitoring stack
start_monitoring() {
    log "Starting monitoring stack..."
    
    # Start main services first
    docker-compose up -d postgres redis vislang-app
    
    # Wait a bit for services to start
    sleep 10
    
    # Start monitoring services
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
    
    # Start monitoring profile services
    docker-compose --profile monitoring up -d
    
    success "Monitoring stack started"
    
    # Show service URLs
    log "Monitoring services are available at:"
    echo "  - Grafana: http://localhost:3000 (admin/$GRAFANA_ADMIN_PASSWORD)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Alertmanager: http://localhost:9093"
    echo "  - cAdvisor: http://localhost:8080"
}

# Setup monitoring alerts
setup_alerts() {
    log "Setting up monitoring alerts..."
    
    # Create alert test script
    cat > scripts/monitoring/test-alerts.sh << 'EOF'
#!/bin/bash
# Test monitoring alerts

echo "Testing alert endpoints..."

# Test Prometheus
echo -n "Prometheus: "
curl -s http://localhost:9090/-/healthy && echo "OK" || echo "FAILED"

# Test Alertmanager
echo -n "Alertmanager: "
curl -s http://localhost:9093/-/healthy && echo "OK" || echo "FAILED"

# Test Grafana
echo -n "Grafana: "
curl -s http://localhost:3000/api/health && echo "OK" || echo "FAILED"

# Fire test alert
echo "Firing test alert..."
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[
    {
      "labels": {
        "alertname": "TestAlert",
        "service": "test",
        "severity": "warning"
      },
      "annotations": {
        "summary": "This is a test alert",
        "description": "Test alert fired from monitoring setup script"
      },
      "startsAt": "'"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"'",
      "endsAt": "'"$(date -u -d '+1 minute' +%Y-%m-%dT%H:%M:%S.%3NZ)"'"
    }
  ]'

echo "Test alert fired. Check Alertmanager UI and configured notification channels."
EOF
    
    chmod +x scripts/monitoring/test-alerts.sh
    success "Alert testing script created"
}

# Generate monitoring documentation
generate_documentation() {
    log "Generating monitoring documentation..."
    
    cat > docs/monitoring/MONITORING.md << EOF
# Monitoring Setup

This document describes the monitoring setup for VisLang-UltraLow-Resource.

## Components

### Prometheus
- **URL**: http://localhost:9090
- **Purpose**: Metrics collection and alerting
- **Configuration**: \`monitoring/prometheus.yml\`

### Grafana
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: $GRAFANA_ADMIN_PASSWORD
- **Purpose**: Metrics visualization and dashboards

### Alertmanager
- **URL**: http://localhost:9093
- **Purpose**: Alert routing and notifications
- **Configuration**: \`monitoring/alertmanager.yml\`

### Exporters
- **PostgreSQL Exporter**: http://localhost:9187/metrics
- **Redis Exporter**: http://localhost:9121/metrics
- **cAdvisor**: http://localhost:8080

## Alert Rules

Alert rules are defined in \`monitoring/alert_rules.yml\` and include:

- Application availability
- Performance metrics
- Resource usage
- Database health
- ML pipeline status

## Dashboards

Pre-configured Grafana dashboards:
- Application Overview
- Performance Metrics
- Infrastructure Health

## Testing

Run alert tests with:
\`\`\`bash
./scripts/monitoring/test-alerts.sh
\`\`\`

## Troubleshooting

### Common Issues

1. **Grafana not starting**: Check permissions on data directory
2. **No metrics**: Verify exporters are running and accessible
3. **Alerts not firing**: Check Prometheus configuration and rules

### Health Check

Run comprehensive health check:
\`\`\`bash
python scripts/monitoring/health-check.py --format summary
\`\`\`
EOF
    
    success "Monitoring documentation generated"
}

# Main setup function
main() {
    log "Starting monitoring setup for VisLang-UltraLow-Resource"
    
    # Pre-flight checks
    check_dependencies
    
    # Setup monitoring infrastructure
    setup_directories
    setup_grafana_datasources
    setup_grafana_dashboards
    setup_alertmanager
    configure_prometheus_targets
    install_exporters
    setup_log_rotation
    
    # Validate configuration
    validate_configuration
    
    # Setup alerts and testing
    setup_alerts
    
    # Generate documentation
    mkdir -p docs/monitoring
    generate_documentation
    
    success "Monitoring setup completed successfully!"
    
    log "Next steps:"
    echo "1. Start monitoring stack: make monitoring-up"
    echo "2. Access Grafana: http://localhost:3000 (admin/$GRAFANA_ADMIN_PASSWORD)"
    echo "3. Test alerts: ./scripts/monitoring/test-alerts.sh"
    echo "4. Run health check: python scripts/monitoring/health-check.py"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --monitoring-dir)
            MONITORING_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --grafana-password)
            GRAFANA_ADMIN_PASSWORD="$2"
            shift 2
            ;;
        --start)
            START_SERVICES=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --monitoring-dir DIR    Monitoring configuration directory"
            echo "  --data-dir DIR          Data directory for persistent storage"
            echo "  --grafana-password PWD  Grafana admin password"
            echo "  --start                 Start monitoring services after setup"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main

# Start services if requested
if [[ "${START_SERVICES:-false}" == "true" ]]; then
    start_monitoring
fi