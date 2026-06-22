# Build and push InsightLab API image to Amazon ECR.
#
# From insight-lab repo root (PowerShell):
#   .\scripts\push-ecr.ps1
#
# Or with explicit vars:
#   $env:ECR_REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
#   $env:ECR_REPOSITORY = "insight-lab"
#   .\scripts\push-ecr.ps1

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$RepoName = if ($env:ECR_REPOSITORY) { $env:ECR_REPOSITORY } else { "insight-lab" }
$Tag = if ($env:API_IMAGE_TAG) { $env:API_IMAGE_TAG } else { "api-latest" }

if (-not $env:ECR_REGISTRY) {
    $AccountId = aws sts get-caller-identity --query Account --output text
    if (-not $AccountId) { throw "Could not resolve AWS account. Run aws configure or set ECR_REGISTRY." }
    $env:ECR_REGISTRY = "$AccountId.dkr.ecr.$Region.amazonaws.com"
}

$Image = "$($env:ECR_REGISTRY)/${RepoName}:$Tag"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "Logging in to ECR: $($env:ECR_REGISTRY) ($Region)"
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $env:ECR_REGISTRY

$repoExists = aws ecr describe-repositories --repository-names $RepoName --region $Region 2>$null
if (-not $repoExists) {
    Write-Host "Creating ECR repository: $RepoName"
    aws ecr create-repository --repository-name $RepoName --region $Region | Out-Null
}

Write-Host "Building API image from backend/Dockerfile..."
docker build -f backend/Dockerfile -t insightlab-api:local backend

docker tag insightlab-api:local $Image

Write-Host "Pushing $Image ..."
docker push $Image

Write-Host ""
Write-Host "Done. On EC2:"
Write-Host "  docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr pull"
Write-Host "  docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d"
