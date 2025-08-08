# Project Configuration
variable "project_name" {
  description = "Name of the project, used for resource naming"
  type        = string
  default     = "movie-prediction-mlops"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.aws_region))
    error_message = "AWS region must be a valid region identifier."
  }
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]

  validation {
    condition     = length(var.availability_zones) >= 2
    error_message = "At least 2 availability zones must be specified for high availability."
  }
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]

  validation {
    condition     = length(var.public_subnet_cidrs) >= 2
    error_message = "At least 2 public subnets must be specified."
  }
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]

  validation {
    condition     = length(var.private_subnet_cidrs) >= 2
    error_message = "At least 2 private subnets must be specified."
  }
}

# EC2 Configuration
variable "instance_type" {
  description = "EC2 instance type for application servers"
  type        = string
  default     = "t2.micro"

  validation {
    condition     = contains(["t2.micro", "t2.small", "t3.micro", "t3.small"], var.instance_type)
    error_message = "Instance type must be Free Tier eligible or cost-effective option."
  }
}

variable "min_size" {
  description = "Minimum number of instances in Auto Scaling Group"
  type        = number
  default     = 1

  validation {
    condition     = var.min_size >= 1 && var.min_size <= 3
    error_message = "Min size must be between 1 and 3 for cost control."
  }
}

variable "max_size" {
  description = "Maximum number of instances in Auto Scaling Group"
  type        = number
  default     = 2

  validation {
    condition     = var.max_size >= var.min_size && var.max_size <= 5
    error_message = "Max size must be >= min_size and <= 5 for cost control."
  }
}

variable "desired_capacity" {
  description = "Desired number of instances in Auto Scaling Group"
  type        = number
  default     = 1

  validation {
    condition     = var.desired_capacity >= var.min_size && var.desired_capacity <= var.max_size
    error_message = "Desired capacity must be between min_size and max_size."
  }
}

variable "public_key" {
  description = "Public key for EC2 key pair (for SSH access)"
  type        = string
  sensitive   = true

  validation {
    condition     = can(regex("^ssh-", var.public_key))
    error_message = "Public key must be in valid SSH public key format."
  }
}

# RDS Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t2.micro"

  validation {
    condition     = contains(["db.t2.micro", "db.t3.micro"], var.db_instance_class)
    error_message = "DB instance class must be Free Tier eligible."
  }
}

variable "db_name" {
  description = "Name of the MLflow database"
  type        = string
  default     = "mlflow"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.db_name))
    error_message = "Database name must start with a letter and contain only alphanumeric characters and underscores."
  }
}

variable "db_username" {
  description = "Username for the MLflow database"
  type        = string
  default     = "mlflow_user"
  sensitive   = true

  validation {
    condition     = length(var.db_username) >= 4 && length(var.db_username) <= 16
    error_message = "Database username must be between 4 and 16 characters."
  }
}

variable "db_password" {
  description = "Password for the MLflow database"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_password) >= 8
    error_message = "Database password must be at least 8 characters long."
  }
}

variable "db_allocated_storage" {
  description = "Allocated storage for RDS instance (GB)"
  type        = number
  default     = 20

  validation {
    condition     = var.db_allocated_storage >= 20 && var.db_allocated_storage <= 100
    error_message = "Allocated storage must be between 20GB (Free Tier minimum) and 100GB for cost control."
  }
}

variable "db_backup_retention_period" {
  description = "Number of days to retain database backups"
  type        = number
  default     = 7

  validation {
    condition     = var.db_backup_retention_period >= 0 && var.db_backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

# S3 Configuration
variable "s3_bucket_prefix" {
  description = "Prefix for S3 bucket name (bucket will be: prefix-mlflow-artifacts-random)"
  type        = string
  default     = "movie-prediction"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.s3_bucket_prefix))
    error_message = "S3 bucket prefix must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "enable_s3_versioning" {
  description = "Enable versioning for S3 bucket"
  type        = bool
  default     = true
}

variable "enable_s3_encryption" {
  description = "Enable server-side encryption for S3 bucket"
  type        = bool
  default     = true
}

# Application Configuration
variable "app_port" {
  description = "Port for the inference application"
  type        = number
  default     = 8000

  validation {
    condition     = var.app_port > 1024 && var.app_port < 65536
    error_message = "Application port must be between 1024 and 65535."
  }
}

variable "monitoring_port" {
  description = "Port for the monitoring application"
  type        = number
  default     = 9000

  validation {
    condition     = var.monitoring_port > 1024 && var.monitoring_port < 65536 && var.monitoring_port != var.app_port
    error_message = "Monitoring port must be between 1024 and 65535 and different from app port."
  }
}

variable "health_check_path" {
  description = "Health check path for load balancer"
  type        = string
  default     = "/health"

  validation {
    condition     = can(regex("^/", var.health_check_path))
    error_message = "Health check path must start with '/'."
  }
}

# Security Configuration
variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application (restrict for production)"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Allow all - restrict in production

  validation {
    condition     = length(var.allowed_cidr_blocks) > 0
    error_message = "At least one CIDR block must be specified."
  }
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = false # Set to true for production
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "sns_email_endpoint" {
  description = "Email address for SNS notifications"
  type        = string
  default     = ""
  sensitive   = true

  validation {
    condition     = var.sns_email_endpoint == "" || can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.sns_email_endpoint))
    error_message = "SNS email endpoint must be a valid email address or empty."
  }
}

# Cost Control
variable "enable_auto_shutdown" {
  description = "Enable automatic shutdown during non-business hours"
  type        = bool
  default     = true
}

variable "business_hours_start" {
  description = "Business hours start time (24-hour format)"
  type        = string
  default     = "09:00"

  validation {
    condition     = can(regex("^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", var.business_hours_start))
    error_message = "Business hours start must be in HH:MM format."
  }
}

variable "business_hours_end" {
  description = "Business hours end time (24-hour format)"
  type        = string
  default     = "18:00"

  validation {
    condition     = can(regex("^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$", var.business_hours_end))
    error_message = "Business hours end must be in HH:MM format."
  }
}

# Docker Configuration
variable "docker_image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"

  validation {
    condition     = length(var.docker_image_tag) > 0
    error_message = "Docker image tag cannot be empty."
  }
}

variable "ecr_repository_url" {
  description = "ECR repository URL for Docker images"
  type        = string
  default     = ""
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default = {
    Owner      = "mlops-team"
    Purpose    = "movie-prediction"
    CostCenter = "ml-ops"
  }
}
