#!/bin/bash

if [ -f ./data/n$1_t$2.csv ]; then
    echo "File exists."
else
    echo "File does not exist."
fi

echo /data/n$1_t$2.csv