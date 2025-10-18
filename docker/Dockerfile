# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PIP_DEFAULT_TIMEOUT=180            

SHELL ["/bin/bash", "-o", "pipefail", "-eEu", "-c"]

# ───────── 0. base OS ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata locales lsb-release ca-certificates gnupg2 \
      build-essential dkms curl git cmake ninja-build \
      python3 python3-dev python3-pip python-is-python3 \
      libeigen3-dev libopenblas-dev liblapack-dev gfortran \
      libboost-all-dev libgl1-mesa-glx libglib2.0-0 \
      libjpeg-dev libpng-dev libopencv-dev \
      libglfw3 libglfw3-dev && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone && \
    locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

# ───────── 1. CUDA 11.3.1 runtime & toolkit ─────────────────────────────────
ENV PATH=/usr/local/cuda-11.3/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64
RUN curl -fsSL -o cuda.run \
      https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run && \
    sh cuda.run --silent --toolkit --override && rm cuda.run

# ───────── 2. ROS 2 Humble desktop ──────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg arch=$(dpkg --print-architecture)] \
     http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    >/etc/apt/sources.list.d/ros2.list
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-desktop ros-humble-cv-bridge ros-humble-image-transport \
      ros-humble-camera-info-manager ros-humble-message-filters \
      python3-colcon-common-extensions && \
    rosdep init || true && rosdep update || true && rm -rf /var/lib/apt/lists/*

# ───────── 3. upgrade pip tools ─────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir --upgrade \
      pip==24.2 setuptools==69.5.1 wheel==0.41.2 packaging==23.2

# ───────── 4. requirements (HGGD README) ────────────────────────────────────
RUN python3 -m pip install --no-cache-dir --retries 5 \
      numpy==1.23.5 \
      pandas \
      scikit-image==0.23.2 \
      matplotlib==3.7.2 \
      'contourpy<1.2' \
      'kiwisolver<1.4.10' \
      pillow==9.5.0 \
      cycler==0.11.0 \
      fonttools \
      pyparsing \
      'python-dateutil<2.9' \
      six \
      tqdm && \
    \
    python3 -m pip install --no-cache-dir --retries 5 \
      --extra-index-url https://download.pytorch.org/whl/cu113 \
      torch==1.11.0+cu113 \
      torchvision==0.12.0+cu113 \
      torchaudio==0.11.0 && \
    \
    python3 -m pip install --no-cache-dir --retries 5 \
      -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1110/download.html \
      pytorch3d && \
    \
    python3 -m pip install --no-cache-dir --retries 5 \
      cupy-cuda11x==12.0.0 \
      cupoch==0.2.11.0 \
      open3d==0.19.0 \
      numba==0.60.0 \
      grasp_nms==1.0.2 \
      opencv-python==4.8.0.76 \
      tensorboardX==2.6.2 \
      torchsummary==1.5.1 \
      transforms3d==0.4.1 \
      trimesh==4.4.0 \
      autolab_core==1.1.1 \   
      cvxopt==1.3.2 \
      fvcore==0.1.5.post20221221 \
      iopath==0.1.10 \
      yacs==0.1.8 \
      termcolor==2.4.0 \
      tabulate==0.9.0 \
      pyyaml==6.0.1 \
      portalocker==2.8.2



# ───────── 5. clone the repo (read-only; bind-mount overrides) ──────────────
RUN git clone --depth 1 https://github.com/THU-VCLab/HGGD.git /opt/HGGD

# ───────── 6. workspace & ROS env ───────────────────────────────────────────
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws/src
ENV ROS_DISTRO=humble
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc

CMD ["bash"]
