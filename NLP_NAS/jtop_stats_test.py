import os
import time
from threading import Thread
from jtop_stats import jtop_stats

with jtop_stats.JtopStats() as stats:
    # Start continuous delta calculation in a separate thread
    delta_thread = Thread(target=stats.calculate_deltas_periodically, args=(30,))
    delta_thread.start()

    for i in range(90):
        print(f'Waiting for delta calculation... {i}s')
        time.sleep(1)
        os.system('clear')

    # Dump stats
    stats.stop_thread = True
    delta_thread.join()
    stats.dump_deltas(f'report.txt', f'raw.txt')