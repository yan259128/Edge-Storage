FROM ubuntu:bionic

RUN apt-get update \
 && apt-get install gnupg -y

RUN apt-get install software-properties-common -y \
 && apt-get update

RUN echo "deb [arch=amd64] http://repo.sawtooth.me/ubuntu/nightly bionic universe" >> /etc/apt/sources.list \
 && (apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 44FC67F19B2466EA \
 || apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 44FC67F19B2466EA \
 || apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 8AA7AF1F1091A5FD)

RUN apt-get update \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y -q \
    python3.7 \
    python3-pip \
    python3-setuptools

RUN mkdir /usr/lib/python3.7/site-packages \
 && cp /usr/lib/python3/dist-packages/apt_pkg.cpython-36m-x86_64-linux-gnu.so  /usr/lib/python3.7/site-packages/apt_pkg.cpython-37m-x86_64-linux-gnu.so

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN apt-get -y install libzmq3-dev \
 && apt-get remove python3-zmq -y \
 && pip3 install pyzmq --only-binary=:all: \
 && pip3 install cython



RUN apt-get install libsecp256k1-dev autoconf libtool pkg-config libffi-dev -y
RUN pip3 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ \
 && apt-get remove python3-secp256k1 -y \
 && pip3 install --no-cache-dir --force-reinstall secp256k1 -i https://mirrors.aliyun.com/pypi/simple/

RUN pip3 install sawtooth_sdk -i https://mirrors.aliyun.com/pypi/simple/ \
 && pip3 install importlib_metadata  -i https://mirrors.aliyun.com/pypi/simple/ \
 && pip3 install protobuf==3.20.0 -i https://mirrors.aliyun.com/pypi/simple/ \
 && pip3 install requests -i https://mirrors.aliyun.com/pypi/simple/

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y apt-utils


RUN apt-get install sawtooth -y

RUN apt-get install curl -y \
    && apt-get install jq -y \
    && apt-get install bc -y \
    && apt-get install procps -y \
    && apt-get install iftop -y \
    && apt-get install coreutils -y

RUN apt-get install -y ifstat \
    && apt-get install vim -y




RUN mkdir -p /var/log/sawtooth

RUN mkdir -p /project/

RUN mkdir -p /etc/sawtooth/keys/

RUN mkdir -p /root/.sawtooth/keys

COPY . /project/
COPY . /project/
COPY ./sawtooth_pote/scripts/testing/. /project/

WORKDIR /project/
