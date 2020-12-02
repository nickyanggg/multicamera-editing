FROM ubuntu:18.04

RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y \
		python3-pip \
		python3-opencv \
		python3-scipy \
		python3-matplotlib \
		libsndfile1 \
		ffmpeg \
	&& dpkg-reconfigure --frontend noninteractive tzdata \
	&& rm -rf /var/lib/apt/lists/*
RUN mkdir code
ADD . /code
WORKDIR /code
RUN pip3 install --upgrade pip
RUN pip3 install --user --upgrade tensorflow-gpu
RUN pip3 install --user --upgrade tensorboard
RUN pip3 install keras==2.3.1
RUN pip3 install --user --upgrade tensorflow-gpu==1.14.0
RUN pip3 install -r requirements.txt
