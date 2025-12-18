#!/bin/bash
export PYTHONPATH="../source:..:$PYTHONPATH"

python ../run_simulation.py param.txt
