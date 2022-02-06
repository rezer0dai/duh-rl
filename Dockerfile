# A Dockerfile that sets up a full Gym install with test dependencies
FROM python:3.8.6
#FROM ubuntu:15.04
RUN apt-get -y update && apt-get install -y python unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg cmake vim

# Get Mujoco
RUN \ 
# Download mujoco
    mkdir /root/.mujoco && \
    cd /root/.mujoco  && \
    curl -O https://www.roboti.us/download/mjpro150_linux.zip  && \
    unzip mjpro150_linux.zip

ADD mjkey.txt /root/.mujoco/
ADD libmujoco150.so /root/.mujoco/mjpro150/bin

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco150/bin:${LD_LIBRARY_PATH}

COPY . /usr/local/
WORKDIR /usr/local/

RUN pip install opencv_python
RUN pip install imageio
RUN pip install cffi
RUN pip install pyglet
RUN pip install lockfile
RUN pip install glfw
RUN pip install numpy
RUN pip install scipy
RUN pip install cython
RUN pip install timebudget
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install gym[mujoco,robotics]

