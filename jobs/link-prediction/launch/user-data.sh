#!/bin/bash
VERSION=0.2.3
SERVER=http://88.99.13.15
touch /home/ubuntu/log-0
export DEBIAN_FRONTEND=noninteractive
touch /home/ubuntu/log-1
apt-get update
touch /home/ubuntu/log-2
apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
touch /home/ubuntu/log-3
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
touch /home/ubuntu/log-4
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
touch /home/ubuntu/log-5
apt-get update
touch /home/ubuntu/log-6
apt-get install -y docker-ce docker-ce-cli containerd.io
touch /home/ubuntu/log-7
curl --url "https://s3.amazonaws.com/public.qwbgewvdrr6sq2eaemcton9wsbqxhzgy/grl-jobs-link-prediction-${VERSION}.tar.gz" --output /home/ubuntu/grl-jobs-link-prediction-${VERSION}.tar.gz
touch /home/ubuntu/log-8
unpigz /home/ubuntu/grl-jobs-link-prediction-${VERSION}.tar.gz
touch /home/ubuntu/log-9
docker load --input /home/ubuntu/grl-jobs-link-prediction-${VERSION}.tar
touch /home/ubuntu/log-10
while true; do docker run grl-jobs-link-prediction:${VERSION} worker --server $SERVER --config /app/worker/config.json; done
