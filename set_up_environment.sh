#!/bin/sh

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt-get install  coinor-cbc coinor-libcbc-dev