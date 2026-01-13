#!/bin/bash
#export PYTHONPATH="../source:..:$PYTHONPATH"
export PYTHONPATH="../../source/package:..:$PYTHONPATH"

#python ../run_simulation.py param.txt


#nr_threads=8
nr_threads=1

#export HPX_NUM_THREADS=$nr_threads

start=$(date +%s)

# python -m cProfile -o profile-original.dat ../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt
python ../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt

end=$(date +%s)
runtime=$((end - start))


echo "Runtime: ${runtime} seconds"
