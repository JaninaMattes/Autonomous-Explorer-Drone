FROM python:3.9

RUN apt-get -y update && apt-get -y install ffmpeg
RUN apt-get -y update && apt-get -y install git wget python-dev python3-dev python-pip zlib1g-dev cmake python-opencv

RUN mkdir -p /usr/src/ppo
WORKDIR /usr/src/ppo
COPY ./gym_pybullet_drones /usr/src/ppo

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    pip install --upgrade pip && \
    pip install torch

RUN cd gym-pybullet-drones/  && \
    pip install -e .
RUN cd gym-pybullet-drones/gym_pybullet_drones/examples/
RUN 
CMD /bin/bash