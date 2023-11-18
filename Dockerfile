FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive 

# dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
  build-essential python3-setuptools \
  libcurl4-openssl-dev python3-dev libssl-dev \
  python3-pip \
 && rm -rf /var/lib/apt/lists/*

# petitor
WORKDIR /opt/petitor
COPY ./requirements.txt ./
RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install -r requirements.txt

# utils
RUN apt-get update \
 && apt-get install -y --no-install-recommends ipython3 iputils-ping iproute2 netcat curl rsh-client telnet vim mlocate nmap \
 && rm -rf /var/lib/apt/lists/* \
 && echo 'set bg=dark' > /root/.vimrc

COPY ./petitor.py ./
ENTRYPOINT ["python3", "./petitor.py"]
