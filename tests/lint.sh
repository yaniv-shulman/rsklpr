#!/bin/bash

while getopts ":f" option; do
   case $option in
      f) # display Help
         FIX=1
         ;;
     \?) # Invalid option
         echo "Error: Invalid option, use -f to fix fixable reported problems"
         exit;;
   esac
done


if [ -z "$FIX" ]; then
    python -m black --check "$REPO_DIR"
    python -m ruff check "$REPO_DIR"
else
    python -m black "$REPO_DIR"
    python -m ruff check "$REPO_DIR" --fix
fi

python -m mypy "$REPO_DIR"
