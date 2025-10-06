#!/usr/bin/env sh

# Simple healthcheck: 0 if /ping responds HTTP 200 quickly
# Requires curl in the image

HOST="127.0.0.1"
PORT="5050"
PATH_="/ping"

curl -fsS -m 0.1 -o /dev/null "http://${HOST}:${PORT}${PATH_}"
