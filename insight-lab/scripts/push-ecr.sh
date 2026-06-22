#!/usr/bin/env bash
# Build and push InsightLab API image to Amazon ECR.
#
# From insight-lab repo root:
#   ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
#   ECR_REPOSITORY=insight-lab \
#   ./scripts/push-ecr.sh
#
# Optional:
#   API_IMAGE_TAG=api-latest
#   AWS_REGION=us-east-1

set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:?Set ECR_REGISTRY (e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com)}"
ECR_REPOSITORY="${ECR_REPOSITORY:-insight-lab}"
AWS_REGION="${AWS_REGION:-us-east-1}"
API_IMAGE_TAG="${API_IMAGE_TAG:-api-latest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY:$API_IMAGE_TAG"

echo "Logging in to ECR: $ECR_REGISTRY ($AWS_REGION)"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Creating ECR repository: $ECR_REPOSITORY"
  aws ecr create-repository --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION"
fi

echo "Building API image from backend/Dockerfile..."
docker build -f backend/Dockerfile -t insightlab-api:local backend

docker tag insightlab-api:local "$IMAGE"

echo "Pushing $IMAGE ..."
docker push "$IMAGE"

echo ""
echo "Done. On EC2, set .env.ecr and run:"
echo "  docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr pull"
echo "  docker compose -f docker-compose.ecr.yml --env-file backend/.env --env-file .env.ecr up -d"
