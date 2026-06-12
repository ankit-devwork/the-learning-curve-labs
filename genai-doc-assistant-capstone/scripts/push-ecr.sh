#!/usr/bin/env bash
# Build, tag, and push backend + Streamlit images to a single ECR repository.
#
# From monorepo root:
#   ECR_REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com \
#   ECR_REPOSITORY=digital-worker-studio \
#   ./genai-doc-assistant-capstone/scripts/push-ecr.sh
#
# Optional tags (defaults shown):
#   BACKEND_IMAGE_TAG=genai-backend-latest
#   STREAMLIT_IMAGE_TAG=genai-streamlit-latest

set -euo pipefail

ECR_REGISTRY="${ECR_REGISTRY:?Set ECR_REGISTRY (e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com)}"
ECR_REPOSITORY="${ECR_REPOSITORY:-digital-worker-studio}"
AWS_REGION="${AWS_REGION:-us-east-1}"
BACKEND_IMAGE_TAG="${BACKEND_IMAGE_TAG:-genai-backend-latest}"
STREAMLIT_IMAGE_TAG="${STREAMLIT_IMAGE_TAG:-genai-streamlit-latest}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BACKEND_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY:$BACKEND_IMAGE_TAG"
STREAMLIT_IMAGE="$ECR_REGISTRY/$ECR_REPOSITORY:$STREAMLIT_IMAGE_TAG"

echo "Logging in to ECR: $ECR_REGISTRY ($AWS_REGION)"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

if ! aws ecr describe-repositories --repository-names "$ECR_REPOSITORY" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Creating ECR repository: $ECR_REPOSITORY"
  aws ecr create-repository --repository-name "$ECR_REPOSITORY" --region "$AWS_REGION"
fi

echo "Building backend image..."
docker build -f genai-doc-assistant-capstone/Dockerfile -t genai-doc-assistant-backend:local .

echo "Building Streamlit image..."
docker build -f genai-doc-assistant-capstone/front-end/streamlit/Dockerfile -t genai-doc-assistant-streamlit:local .

docker tag genai-doc-assistant-backend:local "$BACKEND_IMAGE"
docker tag genai-doc-assistant-streamlit:local "$STREAMLIT_IMAGE"

echo "Pushing backend..."
docker push "$BACKEND_IMAGE"

echo "Pushing Streamlit..."
docker push "$STREAMLIT_IMAGE"

echo ""
echo "Done. Images pushed to repo '$ECR_REPOSITORY':"
echo "  $BACKEND_IMAGE"
echo "  $STREAMLIT_IMAGE"
