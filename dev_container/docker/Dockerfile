FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git sudo zsh nano nvtop htop bash libaio-dev python3-dev && apt-get install -y build-essential python3 python3-pip \
    && apt-get clean \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y jupyter-core \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd ${USER_NAME} --gid ${USER_ID} \
    && useradd -d /home/${USER_NAME} -l -m ${USER_NAME} -u ${USER_ID} -g ${USER_ID} -s /bin/zsh \
    && usermod -aG sudo ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && chmod 440 /etc/sudoers

# Set bash as the default shell
ENV SHELL=/bin/bash

# Upgrade pip
RUN pip install --upgrade pip
RUN pip install jupyterlab
RUN apt-get install python3-dev

# Set working directory and permissions
ENV WORK_DIR=/home/jskim/data_js
ENV NOTEBOOK_DIR=${WORK_DIR}
WORKDIR ${WORK_DIR}
# # 작업 디렉토리 설정
# WORKDIR /home/jskim/data_js/test_241226

RUN mkdir -p ${WORK_DIR} && chmod -R 755 ${WORK_DIR} \
    && chown -R ${USER_NAME}:${USER_NAME} ${WORK_DIR}
    
COPY dev_container/docker/requirements.txt dev_container/docker/requirements.txt

# Add error handling for missing requirements.txt
RUN if [ ! -f "dev_container/docker/requirements.txt" ]; then \
        echo "Error: requirements.txt not found. Ensure the file exists before building the image." && exit 1; \
    fi

# COPY . .

RUN pip install --no-cache-dir -r dev_container/docker/requirements.txt

# Change to the created user
USER ${USER_NAME}
ENV PATH=${PATH}:/home/${USER_NAME}/.local/bin
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/home/jskim/data_js"]
