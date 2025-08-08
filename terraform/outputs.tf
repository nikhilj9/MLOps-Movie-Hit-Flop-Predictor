# VPC and Network Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

# Security Group Outputs
output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "web_security_group_id" {
  description = "ID of the web servers security group"
  value       = aws_security_group.web.id
}

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = aws_security_group.database.id
}

# Database Outputs
output "db_instance_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow.endpoint
}

output "db_instance_address" {
  description = "RDS instance address"
  value       = aws_db_instance.mlflow.address
}

output "db_instance_port" {
  description = "RDS instance port"
  value       = aws_db_instance.mlflow.port
}

output "db_instance_name" {
  description = "RDS instance database name"
  value       = aws_db_instance.mlflow.db_name
}

output "db_instance_username" {
  description = "RDS instance username"
  value       = aws_db_instance.mlflow.username
  sensitive   = true
}

output "db_connection_string" {
  description = "Database connection string for MLflow"
  value       = "postgresql://${aws_db_instance.mlflow.username}:${var.db_password}@${aws_db_instance.mlflow.endpoint}/${aws_db_instance.mlflow.db_name}"
  sensitive   = true
}

# S3 Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.bucket_domain_name
}

output "s3_artifacts_uri" {
  description = "S3 URI for MLflow artifacts"
  value       = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}/artifacts"
}

# Load Balancer Outputs
output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.main.arn
}

output "target_group_arn" {
  description = "ARN of the target group"
  value       = aws_lb_target_group.app.arn
}

# Application URLs
output "inference_api_url" {
  description = "URL for the inference API"
  value       = "http://${aws_lb.main.dns_name}/docs"
}

output "monitoring_dashboard_url" {
  description = "URL for the monitoring dashboard"
  value       = "http://${aws_lb.main.dns_name}:9000/dashboard"
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_lb.main.dns_name}/health"
}

# EC2 and Auto Scaling Outputs
output "launch_template_id" {
  description = "ID of the launch template"
  value       = aws_launch_template.app.id
}

output "launch_template_latest_version" {
  description = "Latest version of the launch template"
  value       = aws_launch_template.app.latest_version
}

output "autoscaling_group_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.app.name
}

output "autoscaling_group_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = aws_autoscaling_group.app.arn
}

# IAM Outputs
output "ec2_instance_profile_name" {
  description = "Name of the EC2 instance profile"
  value       = aws_iam_instance_profile.ec2_profile.name
}

output "ec2_instance_profile_arn" {
  description = "ARN of the EC2 instance profile"
  value       = aws_iam_instance_profile.ec2_profile.arn
}

output "ec2_role_name" {
  description = "Name of the EC2 IAM role"
  value       = aws_iam_role.ec2_role.name
}

output "ec2_role_arn" {
  description = "ARN of the EC2 IAM role"
  value       = aws_iam_role.ec2_role.arn
}

# Key Pair Output
output "key_pair_name" {
  description = "Name of the EC2 key pair"
  value       = aws_key_pair.main.key_name
}

output "key_pair_fingerprint" {
  description = "Fingerprint of the key pair"
  value       = aws_key_pair.main.fingerprint
}

# Environment Variables for Applications
output "environment_variables" {
  description = "Environment variables for application deployment"
  value = {
    ENVIRONMENT                  = "cloud"
    AWS_REGION                   = var.aws_region
    DB_HOST                      = aws_db_instance.mlflow.address
    DB_NAME                      = aws_db_instance.mlflow.db_name
    DB_USER                      = aws_db_instance.mlflow.username
    DB_PASSWORD                  = var.db_password
    S3_BUCKET                    = aws_s3_bucket.mlflow_artifacts.bucket
    MLFLOW_TRACKING_URI          = "postgresql://${aws_db_instance.mlflow.username}:${var.db_password}@${aws_db_instance.mlflow.endpoint}/${aws_db_instance.mlflow.db_name}"
    MLFLOW_DEFAULT_ARTIFACT_ROOT = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}/artifacts"
    USE_AWS_MONITORING           = "true"
    MONITORING_TEST_MODE         = "false"
    PREFECT_API_URL              = "http://localhost:4200/api"
  }
  sensitive = true
}

# Cost Monitoring Outputs
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    ec2_instances   = "Free Tier: 750 hours/month for t2.micro"
    rds_database    = "Free Tier: 750 hours/month for db.t2.micro"
    s3_storage      = "Free Tier: 5GB storage, 20,000 GET requests"
    data_transfer   = "Free Tier: 1GB outbound per month"
    load_balancer   = "~$16.20/month (750 hours free tier)"
    cloudwatch      = "Free Tier: 10 custom metrics"
    total_estimated = "$16-25/month (after free tier hours)"
  }
}

# Resource Tags Output
output "resource_tags" {
  description = "Tags applied to all resources"
  value = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.additional_tags
  )
}

# Connection Information for Deployment
output "deployment_info" {
  description = "Information needed for application deployment"
  value = {
    vpc_id            = aws_vpc.main.id
    public_subnet_ids = aws_subnet.public[*].id
    security_group_id = aws_security_group.web.id
    load_balancer_dns = aws_lb.main.dns_name
    s3_bucket         = aws_s3_bucket.mlflow_artifacts.bucket
    db_endpoint       = aws_db_instance.mlflow.endpoint
    instance_profile  = aws_iam_instance_profile.ec2_profile.name
    key_pair_name     = aws_key_pair.main.key_name
  }
  sensitive = true
}

# Monitoring and Alerting Outputs
output "cloudwatch_log_group_name" {
  description = "CloudWatch log group name for applications"
  value       = "/aws/ec2/${var.project_name}"
}

output "sns_topic_arn" {
  description = "SNS topic ARN for alerts (to be created by application)"
  value       = "arn:aws:sns:${var.aws_region}:${data.aws_caller_identity.current.account_id}:${var.project_name}-alerts"
}

# AWS Account Information
data "aws_caller_identity" "current" {}

output "aws_account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  description = "AWS Region"
  value       = var.aws_region
}

# Security Information
output "security_summary" {
  description = "Security configuration summary"
  value = {
    vpc_cidr           = aws_vpc.main.cidr_block
    database_private   = "Database is in private subnets"
    encryption_enabled = "S3 and RDS encryption enabled"
    iam_roles          = "Least privilege IAM roles configured"
    security_groups    = "Restrictive security groups configured"
    ssl_termination    = "Available at load balancer level"
  }
}

# Backup and Recovery Information
output "backup_configuration" {
  description = "Backup and recovery configuration"
  value = {
    rds_backup_retention = "${var.db_backup_retention_period} days"
    rds_backup_window    = aws_db_instance.mlflow.backup_window
    s3_versioning        = var.enable_s3_versioning ? "Enabled" : "Disabled"
    s3_encryption        = var.enable_s3_encryption ? "Enabled" : "Disabled"
    automated_snapshots  = "RDS automated snapshots enabled"
  }
}

# Troubleshooting Information
output "troubleshooting_info" {
  description = "Information for troubleshooting deployments"
  value = {
    ssh_command = "ssh -i /path/to/private/key ec2-user@<instance-ip>"
    log_locations = {
      application_logs = "/var/log/app/"
      system_logs      = "/var/log/messages"
      docker_logs      = "docker logs <container-name>"
    }
    common_ports = {
      ssh           = 22
      http          = 80
      https         = 443
      inference_api = var.app_port
      monitoring    = var.monitoring_port
      postgresql    = 5432
    }
  }
}
