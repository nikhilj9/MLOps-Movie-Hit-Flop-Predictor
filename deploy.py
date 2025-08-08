"""Deployment script for Prefect workflows"""

import asyncio

from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments import Deployment

from src.main import complete_movie_prediction_pipeline


async def create_deployment():
    """Create and apply the deployment"""

    deployment = await Deployment.build_from_flow(
        flow=complete_movie_prediction_pipeline,
        name="movie-prediction-training",
        version="1.0.0",
        work_pool_name="default",
        schedule=CronSchedule(cron="0 2 * * 1", timezone="UTC"),
        tags=["ml-training", "production"],
        description="Scheduled movie hit prediction model training pipeline",
        parameters={},
    )

    # Apply the deployment
    deployment_id = await deployment.apply()
    print(f"Deployment created with ID: {deployment_id}")

    return deployment_id


async def main():
    """Main deployment function"""
    print("Creating Prefect deployment...")
    deployment_id = await create_deployment()
    print(f"Deployment successful: {deployment_id}")
    print("Access Prefect UI at: http://localhost:4200")


if __name__ == "__main__":
    asyncio.run(main())