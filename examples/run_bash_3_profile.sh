#!/bin/bash
#export PYTHONPATH="../source:..:$PYTHONPATH"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
export PYTHONPATH="../../source/package:..:$PYTHONPATH"

#python ../run_simulation.py param.txt


#nr_threads=8
nr_threads=1

#export HPX_NUM_THREADS=$nr_threads

start=$(date +%s)

# python -m cProfile -o profile-original.dat ../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt
# python ../../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt

# python -m cProfile -o profile_inier_itr_[100]_global[40]_deactive_print_1.dat ../../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt
python -m cProfile -o profile_inner_itr_[200]_global_[20]_dh_0.05_only_rhs.dat ../source/script/run_simulation.py --hpx:threads=$nr_threads param_h_10_10_NOT_permafrost_vegetation.txt


end=$(date +%s)
runtime=$((end - start))


echo "Runtime: ${runtime} seconds"
