FROM python:3.9

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    python3-dev \
    python3-pip \
    zlib1g-dev \
    cmake \
    python3-opencv

RUN mkdir -p /usr/src/ppo
COPY ./gym_pybullet_drones /usr/src/ppo
WORKDIR /usr/src/ppo

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir torch

#RUN pip install --no-cache-dir -r requirements.txt

# Register the PyBullet environments
RUN cd gym-pybullet-drones/  && \
    pip install --no-cache-dir -e .

WORKDIR /usr/src/ppo/gym-pybullet-drones/gym_pybullet_drones/examples

CMD /bin/bash
