#!/bin/bash

poetry install
REPO_DIR=$(git rev-parse --show-toplevel)
PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH
poetry shell
