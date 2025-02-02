#!/bin/bash

# Default image name and tag
DEFAULT_USER_NAME=jskim
DEFAULT_IMAGE_NAME="new-sapie-backend-dev"
DEFAULT_TAG="ubuntu22-04_cu121-torch23"
DOCKERFILE_PATH="$(pwd)/dev_container/docker/Dockerfile"


# Command-line argument handling
USER_NAME=${1:-$DEFAULT_USER_NAME}
IMAGE_NAME=${2:-$DEFAULT_IMAGE_NAME}
TAG=${3:-$DEFAULT_TAG}

# Get current user's UID and GID
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Docker build command
docker build \
  --network=host \
  -f "$DOCKERFILE_PATH" \
  --build-arg USER_NAME=$USER_NAME \
  --build-arg USER_ID=$USER_ID \
  --build-arg GROUP_ID=$GROUP_ID \
  -t $IMAGE_NAME:$TAG \
  .  # 명령을 실행하는 디렉토리 기준으로 빌드 컨텍스트 지정(COPY, ADD 명령에서 접근 가능한 호스트 파일의 기준 디렉토리)

# Output build information
echo "Build completed:"
echo "User: $USER_NAME"
echo "Image: $IMAGE_NAME:$TAG"

