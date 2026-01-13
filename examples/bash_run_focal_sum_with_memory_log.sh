#!/bin/bash

run_bash_name=run_bash_focal_sum_profile_2
log_path="log_path_$run_bash_name"
mkdir -p "$log_path"

memory_log="${log_path}/memory_log_$(date +%Y%m%d_%H%M%S).txt"

# Run the main script in background
# nohup sh ${run_bash_name}.sh > ${log_path}/${run_bash_name}.out 2> ${log_path}/${run_bash_name}.err &
bash ${run_bash_name}.sh > ${log_path}/${run_bash_name}.out 2> ${log_path}/${run_bash_name}.err &

main_pid=$!

sleep 10  # allow it to start

echo "=== Resource monitoring started at $(date) ===" >> "$memory_log"
echo "Main PID: $main_pid" >> "$memory_log"
echo "" >> "$memory_log"

# Monitor memory usage while the main process is running
while kill -0 $main_pid 2>/dev/null; do
    echo "=== $(date) ===" >> "$memory_log"
    # ps -u $USER -o pid,%cpu,%mem,command --sort=-%mem >> "$memory_log"

    session_id=$(ps -o sid= -p $main_pid | tr -d ' ')
    # ps -o pid,ppid,%cpu,%mem,rss,command -g $session_id  >> "$memory_log"
    ps -o pid,ppid,%cpu,%mem,rss,command -g $session_id | tee -a "$memory_log"

    echo "" >> "$memory_log"
    sleep 5  # every 10 minutes (change as you like)
done

echo "=== Resource monitoring finished at $(date) ===" >> "$memory_log"
