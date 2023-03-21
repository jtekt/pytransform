FROM python:3.8
WORKDIR /code
ENV HOME /root
ENV TZ=Asia/Tokyo
COPY . /code
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install tzdata
RUN python3 -m pip install -U pip &&\
    python3 -m pip install --no-cache-dir pipenv &&\
    python3 -m pipenv install --dev
# Remove caches of apt
RUN apt-get autoremove -y &&\
    apt-get clean &&\
    rm -rf /usr/local/src/*
