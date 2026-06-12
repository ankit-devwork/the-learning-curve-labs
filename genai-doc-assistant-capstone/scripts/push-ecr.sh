#!/usr/bin/env bash
# Build, tag, and push backend + Streamlit images to AWS ECR.
#
# From monorepo root:
#   ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
#   AWS_REGION=us-east-1 \
#   ./genai-doc-assistant-capstone/scripts/push-ecr.sh
#
# Optional: IMAGE_TAG=v1.0.0

set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:?Set ECR_REGISTRY (e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com)}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "Logging in to ECR: $ECR_REGISTRY ($AWS_REGION)"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

for repo in genai-doc-assistant-backend genai-doc-assistant-streamlit; do
  if ! aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Creating ECR repository: $repo"
    aws ecr create-repository --repository-name "$repo" --region "$AWS_REGION"
  fi
done

echo "Building backend image..."
docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-doc-assistant-backend:local .

echo "Building Streamlit image..."
docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-doc-assistant-streamlit:local .

docker tag genai-doc-assistant-backend:local "$ECR_REGISTRY/genai-doc-assistant-backend:$IMAGE_TAG"
docker tag genai-doc-assistant-streamlit:local "$ECR_REGISTRY/genai-doc-assistant-streamlit:$IMAGE_TAG"

echo "Pushing backend..."
docker push "$ECR_REGISTRY/genai-doc-assistant-backend:$IMAGE_TAG"

echo "Pushing Streamlit..."
docker push "$ECR_REGISTRY/genai-doc-assistant-streamlit:$IMAGE_TAG"

echo ""
echo "Done. Images pushed:"
echo "  $ECR_REGISTRY/genai-doc-assistant-backend:$IMAGE_TAG"
echo "  $ECR_REGISTRY/genai-doc-assistant-streamlit:$IMAGE_TAG"
