#!/bin/bash

python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 -m pytest -v