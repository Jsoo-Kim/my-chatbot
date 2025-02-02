#!/bin/bash


# Log file setup
CURRENT_DATE=$(date +%Y%m%d)
LOGFILE="./new-sapie-backend-dev-$CURRENT_DATE.log"  # Log file for storing container details and runtime information


# Define paths and variables
MOUNT_HOST_PATH='/home/jskim/data_js'
MOUNT_CONTAINER_PATH='/home/jskim/data_js'
IMAGE_NAME='new-sapie-backend-dev'
IMAGE_TAG='ubuntu22-04_cu121-torch23'
FIXED_TOKEN='salt123A'
SERVER_NAME='new-sapie-backend-dev'
NETWORK_NAME='new-sapie-backend-dev'
JUPYTER_HOST_PORT=10001      
JUPYTER_CONTAINER_PORT=8888 
ADDITIONAL_SERVICE_HOST_PORTS="5406-5408"
ADDITIONAL_SERVICE_CONTAINER_PORTS="5406-5408"  



# Check if the network exists
if ! docker network inspect $NETWORK_NAME >/dev/null 2>&1; then
    echo "Error: Network '$NETWORK_NAME' does not exist. Please create the network before running this script." >&2
    exit 1
fi

# Run Docker container with the specified configurations
docker run -d --gpus all --shm-size 32g \
  -p "$JUPYTER_HOST_PORT":"$JUPYTER_CONTAINER_PORT" \
  -p "$ADDITIONAL_SERVICE_HOST_PORTS":"$ADDITIONAL_SERVICE_CONTAINER_PORTS" \
  -v ${MOUNT_HOST_PATH}:${MOUNT_CONTAINER_PATH} \
  --network ${NETWORK_NAME} \
  -e JUPYTER_ENABLE_LAB=yes \
  -e NB_USER=jskim \
  -e NB_UID=1001 \
  -e NB_GID=1001 \
  -e JUPYTER_TOKEN="${FIXED_TOKEN}" \
  --name ${SERVER_NAME} \
  ${IMAGE_NAME}:${IMAGE_TAG} > ${LOGFILE} 2>&1

# Append container ID to the log file
echo "Docker container started with ID: $(docker ps -lq)" >> ${LOGFILE}
echo "JupyterLab is running. Access it with token: ${FIXED_TOKEN}" >> ${LOGFILE}

