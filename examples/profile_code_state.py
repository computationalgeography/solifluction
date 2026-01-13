import pstats
import sys
from pstats import SortKey

# Create a Stats object from the profiler output file.
# sys.argv[1] is the first command-line argument (the .prof file)
p = pstats.Stats(sys.argv[1])

# strip_dirs() removes the full path from filenames in the report
# This makes output cleaner: only module.py:function instead of full path
p.strip_dirs()

# sort_stats() sorts the profiling results.
# Options include:
#   SortKey.CALLS       -> sort by number of calls to the function
#   SortKey.CUMULATIVE  -> sort by total time including subcalls
#   SortKey.TIME        -> sort by internal time in the function
# You can chain strip_dirs() and sort_stats() like:
#   p.strip_dirs().sort_stats(SortKey.CALLS)
p.sort_stats(SortKey.CALLS)
# p.sort_stats(SortKey.CUMULATIVE)


# print_stats() displays the profiling results
# You can pass a number to print only top N results, e.g. print_stats(10)
# p.print_stats(20)
# p.print_stats(50)
p.print_stats()
