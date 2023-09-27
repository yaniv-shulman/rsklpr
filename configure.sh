#!/bin/bash

poetry install  --no-root
REPO_DIR=$(git rev-parse --show-toplevel)
PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export REPO_DIR
export PYTHONPATH
poetry shell
