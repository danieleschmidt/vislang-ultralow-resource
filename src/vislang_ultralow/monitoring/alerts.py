"""Advanced alerting system with multiple notification channels."""

import time
import logging
import threading
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: float
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'metadata': self.metadata,
            'resolved_at': self.resolved_at,
            'acknowledged_at': self.acknowledged_at,
            'acknowledged_by': self.acknowledged_by,
            'escalation_level': self.escalation_level,
            'notification_count': self.notification_count
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    title_template: str
    description_template: str
    cooldown_seconds: int = 300  # 5 minutes
    max_notifications: int = 10
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    suppression_rules: List[Callable[[Alert], bool]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize rule state."""
        self.last_triggered = 0
        self.notification_count = 0


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        if not self.enabled:
            return False
            
        try:
            return await self._send(alert)
        except Exception as e:
            logger.error(f"Failed to send notification via {self.name}: {e}")
            return False
    
    async def _send(self, alert: Alert) -> bool:
        """Implementation-specific send method."""
        raise NotImplementedError


class EmailChannel(NotificationChannel):
    """Email notification channel."""
    
    async def _send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            smtp_config = self.config['smtp']
            recipients = self.config['recipients']
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                if smtp_config.get('use_tls', False):
                    server.starttls()
                
                if smtp_config.get('username') and smtp_config.get('password'):
                    server.login(smtp_config['username'], smtp_config['password'])
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body."""
        return f"""
Alert Details:
==============

Title: {alert.title}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}
Status: {alert.status.value}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Alert ID: {alert.id}
"""


class SlackChannel(NotificationChannel):
    """Slack notification channel."""
    
    async def _send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            webhook_url = self.config['webhook_url']
            
            # Color coding based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",      # Green
                AlertSeverity.WARNING: "#ff9900",   # Orange
                AlertSeverity.ERROR: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#990000"   # Dark Red
            }
            
            # Create Slack message
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "title": alert.title,
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Source",
                            "value": alert.source,
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp)),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        }
                    ],
                    "footer": "VisLang Alert System",
                    "ts": int(alert.timestamp)
                }]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"Slack notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    async def _send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            url = self.config['url']
            headers = self.config.get('headers', {'Content-Type': 'application/json'})
            timeout = self.config.get('timeout', 10)
            
            # Prepare payload
            payload = {
                'alert': alert.to_dict(),
                'timestamp': time.time(),
                'source': 'vislang-alert-system'
            }
            
            # Send webhook
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self):
        self.alerts = {}  # alert_id -> Alert
        self.alert_rules = {}  # rule_name -> AlertRule
        self.notification_channels = {}  # channel_name -> NotificationChannel
        self.alert_history = deque(maxlen=10000)  # Keep last 10k alerts
        self.is_running = False
        self.processing_thread = None
        self.notification_queue = deque()
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_source': defaultdict(int),
            'notifications_sent': 0,
            'notification_failures': 0
        }
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        with self.lock:
            self.notification_channels[channel.name] = channel
            logger.info(f"Added notification channel: {channel.name}")
    
    def create_alert(self, title: str, description: str, severity: AlertSeverity,
                    source: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{source}_{int(time.time() * 1000)}_{hash(title) % 10000}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        with self.lock:
            # Check for duplicate/similar alerts
            existing_alert = self._find_similar_alert(alert)
            if existing_alert:
                logger.debug(f"Similar alert exists, updating: {existing_alert.id}")
                existing_alert.notification_count += 1
                existing_alert.timestamp = alert.timestamp
                return existing_alert
            
            # Add new alert
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_severity'][severity.value] += 1
            self.stats['alerts_by_source'][source] += 1
            
            # Queue for notification
            self.notification_queue.append(alert)
            
            logger.info(f"Created alert: {alert.id} ({severity.value}) - {title}")
            
        return alert
    
    def _find_similar_alert(self, new_alert: Alert) -> Optional[Alert]:
        """Find similar active alert to avoid duplicates."""
        # Look for alerts from same source with same title in last 5 minutes
        cutoff_time = time.time() - 300
        
        for alert in self.alerts.values():
            if (alert.status == AlertStatus.ACTIVE and
                alert.source == new_alert.source and
                alert.title == new_alert.title and
                alert.timestamp > cutoff_time):
                return alert
        
        return None
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge an alert."""
        with self.lock:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert."""
        with self.lock:
            if alert_id not in self.alerts:
                return False
            
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            
            logger.info(f"Alert suppressed: {alert_id}")
            return True
    
    def check_rules(self, metrics: Dict[str, Any]):
        """Check alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check cooldown
                if current_time - rule.last_triggered < rule.cooldown_seconds:
                    continue
                
                # Check condition
                if rule.condition(metrics):
                    # Create alert from rule
                    title = rule.title_template.format(**metrics)
                    description = rule.description_template.format(**metrics)
                    
                    alert = self.create_alert(
                        title=title,
                        description=description,
                        severity=rule.severity,
                        source=f"rule:{rule_name}",
                        metadata={'rule_name': rule_name, 'metrics': metrics}
                    )
                    
                    rule.last_triggered = current_time
                    logger.info(f"Alert rule triggered: {rule_name}")
                
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    def start_processing(self):
        """Start alert processing and notification."""
        if self.is_running:
            logger.warning("Alert processing already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Alert processing started")
    
    def stop_processing(self):
        """Stop alert processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Alert processing stopped")
    
    def _processing_loop(self):
        """Main alert processing loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while self.is_running:
            try:
                # Process notification queue
                while self.notification_queue and self.is_running:
                    alert = self.notification_queue.popleft()
                    loop.run_until_complete(self._process_alert_notifications(alert))
                
                # Check for escalations
                self._check_escalations()
                
                # Cleanup old resolved alerts
                self._cleanup_resolved_alerts()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                time.sleep(5)  # Wait longer on error
    
    async def _process_alert_notifications(self, alert: Alert):
        """Process notifications for an alert."""
        if alert.status != AlertStatus.ACTIVE:
            return
        
        # Check suppression rules
        for rule in self.alert_rules.values():
            for suppression_rule in rule.suppression_rules:
                if suppression_rule(alert):
                    logger.info(f"Alert suppressed by rule: {alert.id}")
                    alert.status = AlertStatus.SUPPRESSED
                    return
        
        # Send notifications
        sent_count = 0
        failed_count = 0
        
        for channel_name, channel in self.notification_channels.items():
            try:
                if await channel.send_notification(alert):
                    sent_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Notification channel {channel_name} failed: {e}")
                failed_count += 1
        
        # Update statistics
        self.stats['notifications_sent'] += sent_count
        self.stats['notification_failures'] += failed_count
        
        # Update alert
        alert.notification_count += 1
        
        logger.info(f"Processed notifications for alert {alert.id}: {sent_count} sent, {failed_count} failed")
    
    def _check_escalations(self):
        """Check for alert escalations."""
        current_time = time.time()
        
        for alert in self.alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
            
            # Check if alert should be escalated
            age_minutes = (current_time - alert.timestamp) / 60
            
            if age_minutes > 30 and alert.escalation_level == 0:  # Escalate after 30 minutes
                alert.escalation_level = 1
                self._escalate_alert(alert)
            elif age_minutes > 120 and alert.escalation_level == 1:  # Escalate after 2 hours
                alert.escalation_level = 2
                self._escalate_alert(alert)
    
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert."""
        escalated_alert = self.create_alert(
            title=f"ESCALATED: {alert.title}",
            description=f"Alert has been escalated (Level {alert.escalation_level})\n\n{alert.description}",
            severity=AlertSeverity.CRITICAL if alert.escalation_level > 1 else AlertSeverity.ERROR,
            source=f"escalation:{alert.source}",
            metadata={**alert.metadata, 'original_alert_id': alert.id, 'escalation_level': alert.escalation_level}
        )
        
        logger.warning(f"Alert escalated: {alert.id} -> {escalated_alert.id} (Level {alert.escalation_level})")
    
    def _cleanup_resolved_alerts(self):
        """Remove old resolved alerts."""
        cutoff_time = time.time() - (24 * 3600)  # 24 hours
        
        with self.lock:
            to_remove = []
            for alert_id, alert in self.alerts.items():
                if (alert.status in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED] and
                    (alert.resolved_at or alert.timestamp) < cutoff_time):
                    to_remove.append(alert_id)
            
            for alert_id in to_remove:
                del self.alerts[alert_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old alerts")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self.lock:
            active_alerts = self.get_active_alerts()
            
            return {
                **self.stats,
                'active_alerts': len(active_alerts),
                'total_stored_alerts': len(self.alerts),
                'alert_rules': len(self.alert_rules),
                'notification_channels': len(self.notification_channels),
                'queue_size': len(self.notification_queue),
                'active_by_severity': {
                    severity.value: sum(1 for a in active_alerts if a.severity == severity)
                    for severity in AlertSeverity
                }
            }
    
    def export_alerts(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Export alerts to dictionary format."""
        with self.lock:
            alerts_to_export = []
            
            for alert in self.alerts.values():
                if not include_resolved and alert.status in [AlertStatus.RESOLVED, AlertStatus.SUPPRESSED]:
                    continue
                alerts_to_export.append(alert.to_dict())
            
            return sorted(alerts_to_export, key=lambda x: x['timestamp'], reverse=True)