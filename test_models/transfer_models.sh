#!/bin/bash
# Transfers all *.h5 files in the current folder and subfolder to the models folder on the SPURV
for file in * **/*; do
    if [[ $file ==   *.h5 ]]; then
        scp ${file} nvidia@spurv:/home/nvidia/ros/src/spurv_research/spurv_examples/src/models/${file}
    fi
done
