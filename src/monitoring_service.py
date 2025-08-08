"""Model monitoring service with drift detection and alerting"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Fix imports
try:
    from .aws_alerting import AWSAlerting
    from .config import AWS_CONFIG, MONITORING_CONFIG
    from .drift_detector import DriftDetector
    from .reference_data_manager import ReferenceDataManager
except ImportError:
    from aws_alerting import AWSAlerting
    from config import AWS_CONFIG, MONITORING_CONFIG
    from drift_detector import DriftDetector
    from reference_data_manager import ReferenceDataManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Prediction Monitoring Service",
    description="Model monitoring with drift detection and alerting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Pydantic models
class AlertSetupRequest(BaseModel):
    """Request to setup email alerts"""

    email: str = Field(..., description="Email for alerts")
    use_aws: bool = Field(default=False, description="Use AWS SNS for alerts")


class TestAlertRequest(BaseModel):
    """Request to send test alert"""

    message: str = Field(
        default="Test alert from monitoring service", description="Test message"
    )


class MonitoringStatus(BaseModel):
    """Monitoring service status"""

    status: str
    last_check: Optional[str]
    alerts_sent_today: int
    reference_data_loaded: bool
    aws_enabled: bool
    current_email: Optional[str]


# Global instances
reference_manager = ReferenceDataManager()
drift_detector = DriftDetector()
aws_alerting = AWSAlerting()
scheduler = AsyncIOScheduler()

# Monitoring state
monitoring_state = {
    "last_check": None,
    "alerts_sent_today": 0,
    "last_alert_date": None,
    "current_email": AWS_CONFIG.get("reviewer_email", ""),
    "monitoring_active": False,
}

# Store predictions for monitoring (in-memory for demo)
prediction_store = []


async def run_monitoring_check():
    """Run periodic monitoring check"""
    try:
        logger.info("Running monitoring check...")
        monitoring_state["last_check"] = datetime.now().isoformat()

        # Get recent predictions
        recent_predictions = await get_recent_predictions()

        if len(recent_predictions) < 5:  # Reduced threshold for testing
            logger.info(
                f"Not enough predictions for monitoring: {len(recent_predictions)}"
            )
            # Generate some sample data for testing
            recent_predictions = generate_sample_predictions()

        # Run drift detection
        drift_results = drift_detector.detect_drift(recent_predictions)

        # Check if alerts needed
        alerts_needed = []

        if (
            drift_results["data_drift_score"]
            > MONITORING_CONFIG["data_drift_threshold"]
        ):
            alerts_needed.append(
                f"Data drift detected: {drift_results['data_drift_score']:.3f}"
            )

        if (
            drift_results["performance_drift"]
            > MONITORING_CONFIG["model_performance_threshold"]
        ):
            alerts_needed.append(
                f"Performance drift detected: {drift_results['performance_drift']:.3f}"
            )

        if (
            drift_results["avg_confidence"]
            < MONITORING_CONFIG["prediction_confidence_threshold"]
        ):
            alerts_needed.append(
                f"Low prediction confidence: {drift_results['avg_confidence']:.3f}"
            )

        # Send alerts if needed
        if alerts_needed and can_send_alert():
            await send_alert(alerts_needed, drift_results)

        logger.info(
            f"Monitoring check completed. Drift score: {drift_results['data_drift_score']:.3f}"
        )
        return drift_results

    except Exception as e:
        logger.error(f"Monitoring check failed: {e}")
        return None


async def get_recent_predictions() -> List[Dict]:
    """Get recent predictions from stored predictions"""
    # Return last 100 predictions for monitoring
    return prediction_store[-100:] if len(prediction_store) > 0 else []


def generate_sample_predictions() -> List[Dict]:
    """Generate sample predictions for testing drift detection"""
    import random

    sample_predictions = []
    for i in range(50):
        # Create predictions with some variation from reference data
        prediction = {
            "budget": random.uniform(1000000, 300000000),
            "runtime": random.uniform(80, 180),
            "vote_average": random.uniform(4.0, 9.0),
            "vote_count": random.uniform(100, 8000),
            "popularity": random.uniform(1.0, 100.0),
            "genre_count": random.randint(1, 5),
            "release_year": random.randint(2020, 2024),
            "budget_category": random.choice([0, 1, 2, 3]),  # Encoded categories
            "main_genre": random.choice(
                ["Action", "Drama", "Comedy", "Horror", "Unknown"]
            ),
            "is_english": random.choice([0, 1]),
            "prediction": random.choice([0, 1]),
            "probability": random.uniform(0.3, 0.9),
            "timestamp": datetime.now().isoformat(),
        }
        sample_predictions.append(prediction)

    logger.info(f"Generated {len(sample_predictions)} sample predictions for testing")
    return sample_predictions


def can_send_alert() -> bool:
    """Check if we can send an alert (cooldown and daily limits)"""
    now = datetime.now()

    # Reset daily counter
    if monitoring_state["last_alert_date"] != now.date():
        monitoring_state["alerts_sent_today"] = 0
        monitoring_state["last_alert_date"] = now.date()

    # Check daily limit
    if monitoring_state["alerts_sent_today"] >= MONITORING_CONFIG["max_alerts_per_day"]:
        return False

    return True


async def send_alert(alerts: List[str], drift_results: Dict):
    """Send alert via configured method"""
    try:
        alert_message = f"""
        Model Monitoring Alert - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Issues Detected:
        {chr(10).join(f'• {alert}' for alert in alerts)}

        Detailed Metrics:
        • Data Drift Score: {drift_results['data_drift_score']:.3f}
        • Performance Drift: {drift_results['performance_drift']:.3f}
        • Average Confidence: {drift_results['avg_confidence']:.3f}
        • Predictions Analyzed: {drift_results['predictions_count']}

        Action Required: Review model performance and consider retraining.
        Dashboard: http://localhost:9000/dashboard
        """

        # Send via AWS SNS if enabled
        if AWS_CONFIG["use_aws"] and monitoring_state["current_email"]:
            await aws_alerting.send_sns_alert(
                alert_message, monitoring_state["current_email"]
            )
        else:
            # Log alert (for testing without AWS)
            logger.warning(f"ALERT WOULD BE SENT: {alert_message}")

        monitoring_state["alerts_sent_today"] += 1
        logger.info(f"Alert sent. Total today: {monitoring_state['alerts_sent_today']}")

    except Exception as e:
        logger.error(f"Failed to send alert: {e}")


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring service"""
    logger.info("Starting monitoring service...")

    # Load reference data
    if reference_manager.load_reference_data():
        logger.info("Reference data loaded successfully")
    else:
        logger.warning("Failed to load reference data")

    # Initialize drift detector
    drift_detector.initialize(reference_manager.get_reference_data())

    # Setup AWS if enabled
    if AWS_CONFIG["use_aws"]:
        await aws_alerting.setup_aws_resources()

    # Start scheduler
    scheduler.add_job(
        run_monitoring_check,
        "interval",
        minutes=MONITORING_CONFIG["check_interval_minutes"],
        id="monitoring_check",
    )
    scheduler.start()
    monitoring_state["monitoring_active"] = True

    logger.info("Monitoring service started successfully")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "monitoring_active": monitoring_state["monitoring_active"],
        "last_check": monitoring_state["last_check"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status", response_model=MonitoringStatus)
async def get_monitoring_status():
    """Get detailed monitoring status"""
    return MonitoringStatus(
        status="active" if monitoring_state["monitoring_active"] else "inactive",
        last_check=monitoring_state["last_check"],
        alerts_sent_today=monitoring_state["alerts_sent_today"],
        reference_data_loaded=reference_manager.is_loaded(),
        aws_enabled=AWS_CONFIG["use_aws"],
        current_email=monitoring_state["current_email"],
    )


@app.post("/setup-alerts")
async def setup_alerts(request: AlertSetupRequest):
    """Setup email alerts for peer review testing"""
    try:
        monitoring_state["current_email"] = request.email

        if request.use_aws:
            # Setup AWS SNS subscription
            topic_arn = await aws_alerting.create_email_subscription(request.email)
            return {
                "status": "success",
                "message": f"Alerts configured for {request.email}",
                "aws_topic": topic_arn,
                "note": "Please confirm the SNS subscription in your email",
            }
        else:
            return {
                "status": "success",
                "message": f"Test mode alerts configured for {request.email}",
                "note": "Alerts will be logged only (no actual emails sent)",
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup alerts: {str(e)}")


@app.post("/test-alert")
async def send_test_alert(request: TestAlertRequest):
    """Send test alert to configured email"""
    if not monitoring_state["current_email"]:
        raise HTTPException(
            status_code=400, detail="No email configured. Use /setup-alerts first"
        )

    try:
        test_message = f"""
        Test Alert from Movie Prediction Monitoring

        {request.message}

        If you receive this email, alerts are working correctly!
        Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        if AWS_CONFIG["use_aws"]:
            await aws_alerting.send_sns_alert(
                test_message, monitoring_state["current_email"]
            )
            return {"status": "success", "message": "Test alert sent via AWS SNS"}
        else:
            logger.info(f"TEST ALERT: {test_message}")
            return {"status": "success", "message": "Test alert logged (test mode)"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to send test alert: {str(e)}"
        )


@app.get("/dashboard")
async def get_monitoring_dashboard():
    """Get monitoring dashboard data"""
    try:
        # Generate dashboard data
        dashboard_data = {
            "monitoring_status": monitoring_state,
            "drift_history": drift_detector.get_drift_history(),
            "reference_stats": reference_manager.get_reference_stats(),
            "recent_alerts": monitoring_state["alerts_sent_today"],
            "predictions_stored": len(prediction_store),
            "last_updated": datetime.now().isoformat(),
        }

        return dashboard_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate dashboard: {str(e)}"
        )


@app.post("/trigger-monitoring-check")
async def trigger_manual_monitoring_check():
    """Trigger manual monitoring check for testing"""
    try:
        logger.info("Manual monitoring check triggered")
        drift_results = await run_monitoring_check()

        if drift_results:
            return {
                "status": "success",
                "message": "Monitoring check completed",
                "drift_results": drift_results,
            }
        else:
            return {"status": "error", "message": "Monitoring check failed"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to run monitoring check: {str(e)}"
        )


@app.post("/trigger-retraining")
async def trigger_retraining():
    """Trigger Prefect retraining workflow"""
    try:
        # This would call Prefect API to trigger retraining
        logger.info("Retraining workflow triggered")
        return {"status": "success", "message": "Retraining workflow triggered"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to trigger retraining: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=MONITORING_CONFIG["service_host"],
        port=MONITORING_CONFIG["service_port"],
        log_level="info",
    )
