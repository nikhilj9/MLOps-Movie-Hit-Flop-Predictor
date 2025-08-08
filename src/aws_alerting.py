"""AWS CloudWatch and SNS integration for alerting"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
except ImportError as e:
    logging.error(f"Boto3 import failed: {e}")
    boto3 = None

# Fix imports
try:
    from .config import AWS_CONFIG, MONITORING_CONFIG
except ImportError:
    from config import AWS_CONFIG, MONITORING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSAlerting:
    """Handles AWS CloudWatch metrics and SNS alerts"""

    def __init__(self):
        self.cloudwatch = None
        self.sns = None
        self.topic_arn = None
        self.aws_enabled = AWS_CONFIG["use_aws"]
        self.region = MONITORING_CONFIG["aws_region"]
        self.namespace = MONITORING_CONFIG["cloudwatch_namespace"]
        self.topic_name = MONITORING_CONFIG["sns_topic_name"]

        if self.aws_enabled and boto3:
            self._initialize_aws_clients()

    def _initialize_aws_clients(self):
        """Initialize AWS clients"""
        try:
            # Initialize CloudWatch client
            self.cloudwatch = boto3.client(
                "cloudwatch",
                region_name=self.region,
                aws_access_key_id=AWS_CONFIG.get("aws_access_key_id"),
                aws_secret_access_key=AWS_CONFIG.get("aws_secret_access_key"),
            )

            # Initialize SNS client
            self.sns = boto3.client(
                "sns",
                region_name=self.region,
                aws_access_key_id=AWS_CONFIG.get("aws_access_key_id"),
                aws_secret_access_key=AWS_CONFIG.get("aws_secret_access_key"),
            )

            logger.info("AWS clients initialized successfully")

        except NoCredentialsError:
            logger.warning("AWS credentials not found. Running in test mode.")
            self.aws_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            self.aws_enabled = False

    async def setup_aws_resources(self) -> bool:
        """Setup AWS CloudWatch and SNS resources"""
        if not self.aws_enabled:
            logger.info("AWS not enabled, skipping resource setup")
            return False

        try:
            # Create SNS topic
            topic_arn = await self._create_sns_topic()
            if topic_arn:
                self.topic_arn = topic_arn
                logger.info(f"AWS resources setup completed. Topic: {topic_arn}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to setup AWS resources: {e}")
            return False

    async def _create_sns_topic(self) -> Optional[str]:
        """Create SNS topic for alerts"""
        try:
            # Check if topic already exists
            topics = self.sns.list_topics()
            for topic in topics.get("Topics", []):
                if self.topic_name in topic["TopicArn"]:
                    logger.info(f"SNS topic already exists: {topic['TopicArn']}")
                    return topic["TopicArn"]

            # Create new topic
            response = self.sns.create_topic(Name=self.topic_name)
            topic_arn = response["TopicArn"]

            # Set topic attributes
            self.sns.set_topic_attributes(
                TopicArn=topic_arn,
                AttributeName="DisplayName",
                AttributeValue="Movie Prediction Monitoring Alerts",
            )

            logger.info(f"Created SNS topic: {topic_arn}")
            return topic_arn

        except Exception as e:
            logger.error(f"Failed to create SNS topic: {e}")
            return None

    async def create_email_subscription(self, email: str) -> Optional[str]:
        """Create email subscription to SNS topic"""
        if not self.aws_enabled or not self.topic_arn:
            logger.warning("AWS not enabled or topic not created")
            return None

        try:
            # Check if subscription already exists
            subscriptions = self.sns.list_subscriptions_by_topic(
                TopicArn=self.topic_arn
            )
            for sub in subscriptions.get("Subscriptions", []):
                if sub.get("Endpoint") == email and sub.get("Protocol") == "email":
                    logger.info(f"Email subscription already exists: {email}")
                    return self.topic_arn

            # Create subscription
            response = self.sns.subscribe(
                TopicArn=self.topic_arn, Protocol="email", Endpoint=email
            )

            subscription_arn = response.get("SubscriptionArn")
            logger.info(f"Created email subscription: {email} -> {subscription_arn}")

            return self.topic_arn

        except Exception as e:
            logger.error(f"Failed to create email subscription: {e}")
            return None

    async def send_sns_alert(self, message: str, email: str = None) -> bool:
        """Send alert via SNS"""
        if not self.aws_enabled:
            logger.info(f"AWS not enabled. Alert would be sent: {message[:100]}...")
            return True

        if not self.topic_arn:
            logger.error("SNS topic not configured")
            return False

        try:
            # Create subscription if email provided and not exists
            if email:
                await self.create_email_subscription(email)

            # Send message
            response = self.sns.publish(
                TopicArn=self.topic_arn,
                Message=message,
                Subject="Monitoring Alert - Movie Prediction System",
            )

            message_id = response.get("MessageId")
            logger.info(f"SNS alert sent successfully: {message_id}")

            # Also log to CloudWatch
            await self._log_alert_to_cloudwatch()

            return True

        except Exception as e:
            logger.error(f"Failed to send SNS alert: {e}")
            return False

    async def send_cloudwatch_metrics(self, drift_results: Dict[str, Any]) -> bool:
        """Send metrics to CloudWatch"""
        if not self.aws_enabled:
            logger.info("AWS not enabled, skipping CloudWatch metrics")
            return True

        try:
            # Prepare metrics data
            metrics_data = [
                {
                    "MetricName": "DataDriftScore",
                    "Value": drift_results.get("data_drift_score", 0.0),
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "PerformanceDrift",
                    "Value": drift_results.get("performance_drift", 0.0),
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "PredictionConfidence",
                    "Value": drift_results.get("avg_confidence", 1.0),
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "DriftDetected",
                    "Value": 1.0 if drift_results.get("drift_detected", False) else 0.0,
                    "Unit": "None",
                    "Timestamp": datetime.now(),
                },
                {
                    "MetricName": "PredictionsAnalyzed",
                    "Value": drift_results.get("predictions_count", 0),
                    "Unit": "Count",
                    "Timestamp": datetime.now(),
                },
            ]

            # Send metrics in batches (CloudWatch limit is 20 metrics per call)
            batch_size = 20
            for i in range(0, len(metrics_data), batch_size):
                batch = metrics_data[i : i + batch_size]

                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace, MetricData=batch
                )

            logger.info(f"Sent {len(metrics_data)} metrics to CloudWatch")
            return True

        except Exception as e:
            logger.error(f"Failed to send CloudWatch metrics: {e}")
            return False

    async def _log_alert_to_cloudwatch(self) -> bool:
        """Log alert event to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[
                    {
                        "MetricName": "AlertsSent",
                        "Value": 1.0,
                        "Unit": "Count",
                        "Timestamp": datetime.now(),
                    }
                ],
            )
            return True

        except Exception as e:
            logger.error(f"Failed to log alert to CloudWatch: {e}")
            return False

    async def create_cloudwatch_dashboard(self) -> Optional[str]:
        """Create CloudWatch dashboard for monitoring"""
        if not self.aws_enabled:
            return None

        try:
            dashboard_name = "MoviePredictionMonitoring"

            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [self.namespace, "DataDriftScore"],
                                [".", "PerformanceDrift"],
                                [".", "PredictionConfidence"],
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.region,
                            "title": "Model Monitoring Metrics",
                        },
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [self.namespace, "DriftDetected"],
                                [".", "AlertsSent"],
                            ],
                            "period": 300,
                            "stat": "Sum",
                            "region": self.region,
                            "title": "Alerts and Drift Detection",
                        },
                    },
                ]
            }

            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name, DashboardBody=json.dumps(dashboard_body)
            )

            dashboard_url = f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard_name}"
            logger.info(f"CloudWatch dashboard created: {dashboard_url}")

            return dashboard_url

        except Exception as e:
            logger.error(f"Failed to create CloudWatch dashboard: {e}")
            return None

    def get_aws_status(self) -> Dict[str, Any]:
        """Get AWS integration status"""
        return {
            "aws_enabled": self.aws_enabled,
            "region": self.region,
            "namespace": self.namespace,
            "topic_arn": self.topic_arn,
            "topic_name": self.topic_name,
            "cloudwatch_available": self.cloudwatch is not None,
            "sns_available": self.sns is not None,
        }

    async def test_aws_connectivity(self) -> Dict[str, Any]:
        """Test AWS connectivity and permissions"""
        if not self.aws_enabled:
            return {"status": "disabled", "message": "AWS integration disabled"}

        test_results = {"cloudwatch": False, "sns": False, "errors": []}

        # Test CloudWatch
        try:
            self.cloudwatch.list_metrics(Namespace=self.namespace, MaxRecords=1)
            test_results["cloudwatch"] = True
        except Exception as e:
            test_results["errors"].append(f"CloudWatch: {str(e)}")

        # Test SNS
        try:
            self.sns.list_topics()
            test_results["sns"] = True
        except Exception as e:
            test_results["errors"].append(f"SNS: {str(e)}")

        test_results["status"] = (
            "success"
            if all([test_results["cloudwatch"], test_results["sns"]])
            else "partial"
        )

        return test_results

    async def cleanup_aws_resources(self) -> bool:
        """Cleanup AWS resources (for testing)"""
        if not self.aws_enabled or AWS_CONFIG.get("test_mode", True):
            logger.info("Skipping cleanup in test mode")
            return True

        try:
            # Delete SNS topic
            if self.topic_arn:
                self.sns.delete_topic(TopicArn=self.topic_arn)
                logger.info(f"Deleted SNS topic: {self.topic_arn}")
                
            # Note: CloudWatch metrics are automatically cleaned up 
            # after retention period

            return True

        except Exception as e:
            logger.error(f"Failed to cleanup AWS resources: {e}")
            return False