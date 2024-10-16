#!/bin/bash

docker-compose down -v

docker-compose create

./create_accounts.sh

python3 generate_genesis.py

./init.sh

