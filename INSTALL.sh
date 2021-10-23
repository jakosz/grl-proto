#!/bin/bash

git pull origin master --tag

python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest -v

sudo mount -o remount,size=100% /dev/shm     # increase the size of shared memory 
sudo mount -o size=100% -t tmpfs tmpfs /tmp  # mount /tmp in RAM
