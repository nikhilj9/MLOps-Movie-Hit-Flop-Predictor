#!/bin/bash
set -e

# Movie Prediction MLOps Pipeline - Cloud Deployment Script
# This script automates the complete deployment to AWS using Terraform

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"
LOG_FILE="$PROJECT_ROOT/deployment.log"
TFVARS_FILE="$TERRAFORM_DIR/terraform.tfvars"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    local missing_tools=()

    # Check for required tools
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi

    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi

    if ! command -v aws &> /dev/null; then
        missing_tools+=("aws-cli")
    fi

    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi

    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        echo "Please install the missing tools and try again."
        echo "Installation commands:"
        echo "  terraform: https://developer.hashicorp.com/terraform/downloads"
        echo "  docker: https://docs.docker.com/get-docker/"
        echo "  aws-cli: pip install awscli"
        echo "  jq: sudo apt-get install jq (Ubuntu) or brew install jq (macOS)"
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        echo "Please run: aws configure"
        exit 1
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        echo "Please start Docker and try again"
        exit 1
    fi

    log "All prerequisites satisfied"
}

# Function to generate SSH key pair
generate_ssh_key() {
    local key_name="movie-prediction-key"
    local key_path="$HOME/.ssh/$key_name"

    if [ -f "$key_path" ]; then
        warning "SSH key already exists at $key_path"
        read -p "Do you want to use the existing key? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Generating new SSH key..."
            rm -f "$key_path" "$key_path.pub"
        else
            return 0
        fi
    fi

    if [ ! -f "$key_path" ]; then
        log "Generating SSH key pair..."
        ssh-keygen -t rsa -b 2048 -f "$key_path" -N "" -C "movie-prediction-mlops"
        chmod 600 "$key_path"
        chmod 644 "$key_path.pub"
        log "SSH key generated at $key_path"
    fi

    # Export public key for use in terraform
    export TF_VAR_public_key=$(cat "$key_path.pub")
    echo "export SSH_PRIVATE_KEY_PATH=\"$key_path\"" >> "$PROJECT_ROOT/.env.deployment"
}

# Function to create terraform.tfvars if it doesn't exist
create_tfvars() {
    if [ ! -f "$TFVARS_FILE" ]; then
        log "Creating terraform.tfvars from example..."

        if [ ! -f "$TERRAFORM_DIR/terraform.tfvars.example" ]; then
            error "terraform.tfvars.example not found"
            exit 1
        fi

        cp "$TERRAFORM_DIR/terraform.tfvars.example" "$TFVARS_FILE"

        # Get AWS account info
        local aws_region=$(aws configure get region || echo "us-east-1")
        local aws_account_id=$(aws sts get-caller-identity --query Account --output text)

        # Update tfvars with dynamic values
        sed -i.bak "s|your-public-key-here|$(cat ~/.ssh/movie-prediction-key.pub)|g" "$TFVARS_FILE"
        sed -i.bak "s|your-email@example.com|$(git config user.email || echo 'admin@example.com')|g" "$TFVARS_FILE"
        sed -i.bak "s|us-east-1|$aws_region|g" "$TFVARS_FILE"

        rm -f "$TFVARS_FILE.bak"

        warning "Please review and customize $TFVARS_FILE before proceeding"
        warning "Especially the database password and email addresses"

        read -p "Press Enter after reviewing terraform.tfvars..." -r
    else
        log "Using existing terraform.tfvars"
    fi
}

# Function to validate terraform configuration
validate_terraform() {
    log "Validating Terraform configuration..."

    cd "$TERRAFORM_DIR"

    # Initialize Terraform
    log "Initializing Terraform..."
    terraform init

    # Validate configuration
    log "Validating Terraform configuration..."
    terraform validate

    # Format check
    terraform fmt -check=true -diff=true || {
        warning "Terraform files need formatting. Running terraform fmt..."
        terraform fmt
    }

    log "Terraform validation completed"
}

# Function to plan deployment
plan_deployment() {
    log "Creating Terraform deployment plan..."

    cd "$TERRAFORM_DIR"

    # Create plan
    terraform plan -var-file="terraform.tfvars" -out="tfplan"

    log "Terraform plan created"
    log "Review the plan above before proceeding with deployment"

    read -p "Do you want to proceed with deployment? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
}

# Function to deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure..."

    cd "$TERRAFORM_DIR"

    # Apply plan
    terraform apply "tfplan"

    # Save outputs to file
    terraform output -json > "$PROJECT_ROOT/terraform-outputs.json"

    log "Infrastructure deployment completed"
}

# Function to build and push Docker images
build_and_push_images() {
    log "Building and pushing Docker images..."

    # Get ECR repository URL from terraform outputs
    local ecr_repository=$(terraform output -raw ecr_repository_url 2>/dev/null || echo "")

    if [ -z "$ecr_repository" ]; then
        warning "ECR repository not found in outputs, skipping image push"
        return 0
    fi

    # Get AWS region
    local aws_region=$(terraform output -raw aws_region)

    # Login to ECR
    log "Logging into ECR..."
    aws ecr get-login-password --region "$aws_region" | docker login --username AWS --password-stdin "$ecr_repository"

    cd "$PROJECT_ROOT"

    # Build inference image
    log "Building inference image..."
    docker build -f Dockerfile.inference -t movie-prediction:latest .
    docker tag movie-prediction:latest "$ecr_repository:latest"
    docker tag movie-prediction:latest "$ecr_repository:$(date +%Y%m%d-%H%M%S)"

    # Push images
    log "Pushing images to ECR..."
    docker push "$ecr_repository:latest"
    docker push "$ecr_repository:$(date +%Y%m%d-%H%M%S)"

    log "Docker images pushed successfully"
}

# Function to wait for deployment readiness
wait_for_deployment() {
    log "Waiting for deployment to be ready..."

    cd "$TERRAFORM_DIR"

    # Get load balancer DNS name
    local lb_dns=$(terraform output -raw load_balancer_dns_name 2>/dev/null || echo "")

    if [ -z "$lb_dns" ]; then
        warning "Load balancer DNS not found, skipping readiness check"
        return 0
    fi

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        log "Checking deployment readiness (attempt $attempt/$max_attempts)..."

        # Check if load balancer is responding
        if curl -s -f "http://$lb_dns/health" > /dev/null 2>&1; then
            log "Deployment is ready"
            return 0
        fi

        sleep 30
        ((attempt++))
    done

    warning "Deployment readiness check timed out"
    warning "The deployment may still be starting up"
}

# Function to run deployment tests
test_deployment() {
    log "Running deployment tests..."

    cd "$TERRAFORM_DIR"

    # Get outputs
    local lb_dns=$(terraform output -raw load_balancer_dns_name 2>/dev/null || echo "")

    if [ -z "$lb_dns" ]; then
        warning "Load balancer DNS not found, skipping tests"
        return 0
    fi

    local test_results=()

    # Test health endpoints
    info "Testing health endpoints..."

    if curl -s -f "http://$lb_dns/health" > /dev/null; then
        test_results+=("Inference API health check: PASS")
    else
        test_results+=("Inference API health check: FAIL")
    fi

    if curl -s -f "http://$lb_dns:9000/health" > /dev/null; then
        test_results+=("Monitoring service health check: PASS")
    else
        test_results+=("Monitoring service health check: FAIL")
    fi

    # Test prediction endpoint
    info "Testing prediction endpoint..."
    local prediction_response=$(curl -s -X POST "http://$lb_dns/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "budget": 50000000,
            "runtime": 120,
            "vote_average": 7.5,
            "vote_count": 1500,
            "popularity": 25.0,
            "genres": "[{\"name\": \"Action\"}]",
            "original_language": "en",
            "release_date": "2023-01-15"
        }' || echo "")

    if echo "$prediction_response" | jq -e '.prediction' > /dev/null 2>&1; then
        test_results+=("Prediction endpoint: PASS")
    else
        test_results+=("Prediction endpoint: FAIL")
    fi

    # Display test results
    log "Test Results:"
    for result in "${test_results[@]}"; do
        echo "  $result"
    done
}

# Function to display deployment information
show_deployment_info() {
    log "Deployment Information:"

    cd "$TERRAFORM_DIR"

    # Get key outputs
    local lb_dns=$(terraform output -raw load_balancer_dns_name 2>/dev/null || echo "N/A")
    local db_endpoint=$(terraform output -raw db_instance_endpoint 2>/dev/null || echo "N/A")
    local s3_bucket=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "N/A")

    echo
    echo "Deployment completed successfully!"
    echo
    echo "Service URLs:"
    echo "  Inference API:       http://$lb_dns/docs"
    echo "  Monitoring Dashboard: http://$lb_dns:9000/dashboard"
    echo "  Health Check:        http://$lb_dns/health"
    echo
    echo "Infrastructure:"
    echo "  Load Balancer:       $lb_dns"
    echo "  Database Endpoint:   $db_endpoint"
    echo "  S3 Artifacts Bucket: $s3_bucket"
    echo
    echo "SSH Access:"
    echo "  Private Key:         ~/.ssh/movie-prediction-key"
    echo "  Connect Command:     ssh -i ~/.ssh/movie-prediction-key ec2-user@<instance-ip>"
    echo
    echo "Important Files:"
    echo "  Terraform Outputs:   $PROJECT_ROOT/terraform-outputs.json"
    echo "  Deployment Log:      $LOG_FILE"
    echo "  Environment Config:  $PROJECT_ROOT/.env.deployment"
    echo
    echo "Cost Estimate: ~$16-25/month (after free tier)"
    echo
    echo "Security Reminders:"
    echo "  - Review security groups for production use"
    echo "  - Change default database password"
    echo "  - Set up proper SSL certificates"
    echo "  - Configure backup retention policies"
    echo
}

# Function to cleanup on failure
cleanup_on_failure() {
    error "Deployment failed!"

    read -p "Do you want to destroy the infrastructure? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Destroying infrastructure..."
        cd "$TERRAFORM_DIR"
        terraform destroy -var-file="terraform.tfvars" -auto-approve
        log "Infrastructure destroyed"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -p, --plan-only     Only create and show the deployment plan"
    echo "  -d, --destroy       Destroy the existing infrastructure"
    echo "  -t, --test-only     Only run deployment tests"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                  Full deployment"
    echo "  $0 --plan-only      Only show deployment plan"
    echo "  $0 --destroy        Destroy infrastructure"
    echo "  $0 --test-only      Test existing deployment"
}

# Function to destroy infrastructure
destroy_infrastructure() {
    log "Destroying infrastructure..."

    cd "$TERRAFORM_DIR"

    if [ ! -f "terraform.tfstate" ]; then
        error "No terraform state found. Nothing to destroy."
        exit 1
    fi

    warning "This will destroy ALL infrastructure resources!"
    read -p "Are you sure you want to continue? (type 'yes' to confirm): " -r

    if [ "$REPLY" != "yes" ]; then
        log "Destruction cancelled"
        exit 0
    fi

    terraform destroy -var-file="terraform.tfvars" -auto-approve

    log "Infrastructure destroyed successfully"
}

# Main execution function
main() {
    local plan_only=false
    local destroy_only=false
    local test_only=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--plan-only)
                plan_only=true
                shift
                ;;
            -d|--destroy)
                destroy_only=true
                shift
                ;;
            -t|--test-only)
                test_only=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Create log file
    touch "$LOG_FILE"

    log "Starting Movie Prediction MLOps Pipeline Deployment"
    log "Timestamp: $(date)"
    log "User: $(whoami)"
    log "Working Directory: $PROJECT_ROOT"

    # Handle different modes
    if [ "$destroy_only" = true ]; then
        destroy_infrastructure
        exit 0
    fi

    if [ "$test_only" = true ]; then
        test_deployment
        exit 0
    fi

    # Set up error handling
    trap cleanup_on_failure ERR

    # Main deployment process
    check_prerequisites
    generate_ssh_key
    create_tfvars
    validate_terraform
    plan_deployment

    if [ "$plan_only" = true ]; then
        log "Plan-only mode completed"
        exit 0
    fi

    deploy_infrastructure
    build_and_push_images
    wait_for_deployment
    test_deployment
    show_deployment_info

    log "Deployment completed successfully!"
}

# Execute main function with all arguments
main "$@"