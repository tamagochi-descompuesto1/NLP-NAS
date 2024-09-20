import time
import numpy as np
from jtop import jtop

class JtopStats:
    def __init__(self)-> None:
        self.power = {}
        self.memory = {}
        self.cpu = {}
        self.gpu = {}
        self.stats = {}
        self.time_window = 3 # For smoothing power and GPU stats
        self.previous_samples = {
            'power': [],
            'gpu_load': []
        }
        self.deltas_history = []  # To store all intermediate deltas
        self.stop_thread = False # Flag to stop the delta calculation thread

    def __enter__(self):
        self.set_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_thread = True # Ensure the thread is stopped when exiting
        pass

    # Getters for attributes
    def get_power(self) -> dict:
        return self.power

    def get_memory(self) -> dict:
        return self.memory

    def get_cpu(self) -> dict:
        return self.cpu

    def get_gpu(self) -> dict:
        return self.gpu

    def get_stats_history(self) -> dict:
        return self.stats

    def __init_time(self, stats: dict) -> dict:
        stats['time'] = time.perf_counter()
        return stats

    def __get_stats(self) -> dict:
        try:
            with jtop() as jetson:
                power = jetson.power.get('tot', {})  # Using .get() to avoid KeyError
                memory = jetson.memory.get('RAM', {})
                cpu = jetson.cpu.get('total', {})
                gpu = jetson.gpu.get('ga10b', {})

                if not power or not memory or not cpu or not gpu:
                    raise ValueError("Incomplete stats retrieved")

                stats = {
                    'power': {
                        'voltage_mV': power.get('volt', 0),
                        'current_mA': power.get('curr', 0),
                        'avg_power_mW': power.get('avg', 0)
                    },
                    'memory': {
                        'used_KB': memory.get('used', 0),
                        'total_KB': memory.get('tot', 0),
                        'free_KB': memory.get('free', 0),
                        'cached_KB': memory.get('cached', 0),
                        'shared_KB': memory.get('shared', 0)
                    },
                    'cpu': {
                        'user': cpu.get('user', 0),
                        'nice': cpu.get('nice', 0),
                        'system': cpu.get('system', 0),
                        'idle': cpu.get('idle', 0),
                        'cpus': [{'cpu': index + 1, 'current_freq': cpu_core.get('freq', {}).get('cur', 0)} for index, cpu_core in enumerate(jetson.cpu.get('cpu', []))]
                    },
                    'gpu': {
                        'load': gpu.get('status', {}).get('load', 0),
                        'min_freq': gpu.get('freq', {}).get('min', 0),
                        'max_freq': gpu.get('freq', {}).get('max', 0),
                        'curr_freq': gpu.get('freq', {}).get('cur', 0)
                    }
                }

                stats = self.__init_time(stats)
                return stats
        except Exception as e:
            print(f"Error fetching stats: {e}")
            return {}

    def set_stats(self) -> None:
        self.stats = self.__get_stats()
        self.power = self.stats['power']
        self.memory = self.stats['memory']
        self.cpu = self.stats['cpu']
        self.gpu = self.stats['gpu']

    def smooth_power(self, samples):
        window_size = self.time_window
        smoothed_samples = []

        for i in range(len(samples)):
            smoothed_samples.append(np.mean(samples[max(0, i - window_size + 1):i + 1]))
        return smoothed_samples

    def smooth_gpu_load(self, samples):
        return self.smooth_power(samples)

    def __get_deltas(self) -> dict:
        if self.stats == {}:
            self.set_stats()

        current_stats = self.__get_stats()
        deltas = {}

        # Smooth power and GPU load to avoid spikes
        self.previous_samples['power'].append(current_stats['power']['avg_power_mW'])
        self.previous_samples['gpu_load'].append(current_stats['gpu']['load'])
        smoothed_power = self.smooth_power(self.previous_samples['power'])
        smoothed_gpu_load = self.smooth_gpu_load(self.previous_samples['gpu_load'])

        # Deltas for power, memory, and GPU
        deltas['power'] = {
            'voltage_mV': current_stats['power']['voltage_mV'] - self.power['voltage_mV'],
            'current_mA': current_stats['power']['current_mA'] - self.power['current_mA'],
            'avg_power_mW': smoothed_power[-1] - self.power['avg_power_mW']
        }

        deltas['memory'] = {key: current_stats['memory'][key] - self.memory[key] for key in current_stats['memory'].keys()}

        deltas['gpu'] = {
            'load': smoothed_gpu_load[-1] - self.gpu['load'],
            'min_freq': current_stats['gpu']['min_freq'] - self.gpu['min_freq'],
            'max_freq': current_stats['gpu']['max_freq'] - self.gpu['max_freq'],
            'curr_freq': current_stats['gpu']['curr_freq'] - self.gpu['curr_freq'],
        }

        # Per-CPU deltas
        deltas['cpu'] = {
            'cpus': []
        }

        # Iterate through each CPU core and calculate the difference in stats
        for index, current_cpu in enumerate(current_stats['cpu']['cpus']):
            previous_cpu = self.cpu['cpus'][index] if index < len(self.cpu['cpus']) else {'current_freq': 0}
            deltas['cpu']['cpus'].append({
                'cpu': index + 1,
                'current_freq': current_cpu['current_freq'] - previous_cpu.get('current_freq', 0)
            })

        # Delta for time
        #deltas['time'] = current_stats['time'] - self.stats['time']

        # Ensure positive deltas where expected
        deltas = self.ensure_positive_deltas(deltas)

        return deltas, current_stats
    
    def calculate_deltas_periodically(self, interval=1):
        """
        Continuously calculate deltas every 'interval' seconds and store them in deltas_history.
        """
        self.set_stats()
        try:
            while not self.stop_thread:
                time.sleep(interval)
                deltas, current_stats = self.__get_deltas()

                if current_stats:  # Proceed only if stats were retrieved successfully
                    self.deltas_history.append(deltas)
                else:
                    print("Warning: Failed to retrieve stats during periodic collection.")
        except KeyboardInterrupt:
            print("Periodic delta collection stopped.")

    def summarize_deltas(self):
        summary = {
            'power': {key: np.mean([delta['power'][key] for delta in self.deltas_history]) for key in self.deltas_history[0]['power'].keys()},
            'memory': {key: np.mean([delta['memory'][key] for delta in self.deltas_history]) for key in self.deltas_history[0]['memory'].keys()},
            'gpu': {key: np.mean([delta['gpu'][key] for delta in self.deltas_history]) for key in self.deltas_history[0]['gpu'].keys()},
            'cpu': {
                'cpus': []
            },
            'time': time.perf_counter() - self.stats['time']
        }

        # Summarize CPU deltas for each core
        num_cpus = len(self.deltas_history[0]['cpu']['cpus'])
        for i in range(num_cpus):
            avg_freq_delta = np.mean([delta['cpu']['cpus'][i]['current_freq'] for delta in self.deltas_history])
            summary['cpu']['cpus'].append({
                'cpu': i + 1,
                'current_freq': avg_freq_delta
            })

        return summary

    
    def ensure_positive_deltas(self, deltas):
        for key in deltas:
            if isinstance(deltas[key], dict):
                # Handle dictionaries
                for stat, value in deltas[key].items():
                    if isinstance(value, list):
                        # If the value is a list (like 'cpu' core stats), iterate through each item
                        for item in value:
                            if isinstance(item, dict):  # Handle the case for lists of dictionaries (like 'cpu')
                                for k, v in item.items():
                                    if isinstance(v, (int, float)) and v < 0:
                                        item[k] = 0
                            elif isinstance(item, (int, float)) and item < 0:
                                item = 0
                    elif isinstance(value, (int, float)) and value < 0:
                        deltas[key][stat] = 0  # Set to 0 if negative
            elif isinstance(deltas[key], (int, float)):
                # Handle floats and integers
                if deltas[key] < 0:
                    deltas[key] = 0  # Set to 0 if negative
        return deltas


    
    def dump_power(self, file, value):
        file.write('POWER:\n')
        file.write('    - Voltage:        ' +  str(value['power']['voltage_mV']) + ' mV\n')
        file.write('    - Current:        ' +  str(value['power']['current_mA']) + ' mA\n')
        file.write('    - Average power:  ' +  str(value['power']['avg_power_mW']) + ' mW\n')
        file.write('\n')

    def dump_memory(self, file, value):
        file.write('MEMORY:\n')
        file.write('    - Used RAM:          ' +  str(value['memory']['used_KB']) + ' KB\n')
        file.write('    - Total RAM:         ' +  str(value['memory']['total_KB']) + ' KB\n')
        file.write('    - Free RAM:          ' +  str(value['memory']['free_KB']) + ' KB\n')
        file.write('    - Cached RAM:        ' +  str(value['memory']['cached_KB']) + ' KB\n')
        file.write('    - Shared RAM:        ' + str(value['memory']['shared_KB']) + ' KB\n')
        file.write('\n')

    def dump_cpu(self, file, value):
        file.write('CPU:\n')
        file.write('    - Total CPU stats:\n')
        
        # Use .get() to provide a default value of 0 if the key is missing
        file.write('        + User utilization:   ' + str(value['cpu'].get('user', 0)) + '\n')
        file.write('        + Nice utilization:   ' + str(value['cpu'].get('nice', 0)) + '\n')
        file.write('        + System utilization: ' + str(value['cpu'].get('system', 0)) + '\n')
        file.write('        + Idle utilization:   ' + str(value['cpu'].get('idle', 0)) + '\n')
        
        # Handle each CPU core stats
        for cpu in value['cpu']['cpus']:
            file.write('    - CPU ' + str(cpu['cpu']) + ' frequency:   ' + str(cpu['current_freq']) + ' kHz\n')
        file.write('\n')


    def dump_gpu(self, file, value):
        file.write('GPU:\n')
        file.write('    - GPU load:                  ' +  str(value['gpu']['load']) + '\n')
        file.write('    - Minimum frequency:         ' +  str(value['gpu']['min_freq']) + ' kHz\n')
        file.write('    - Maximum frequency:         ' + str(value['gpu']['max_freq']) + ' kHz\n')
        file.write('    - Current frequency:         ' +  str(value['gpu']['curr_freq']) + ' kHz\n')
        file.write('\n')

    def dump_deltas(self, report_path: str = '/', raw_data_path: str = '/') -> None:
        deltas, current_stats = self.__get_deltas()
        final_stats = self.summarize_deltas()

        # Writing the report
        with open(report_path, 'w', encoding='utf-8') as report_file:
            for title, value in {'STARTING STATS': self.stats, 'ENDING STATS': current_stats, 'CONSUMPTION STATS': final_stats}.items():
                report_file.write('----------------' + title + '----------------' + '\n')
                if title == 'CONSUMPTION STATS':
                    report_file.write('Total Time:   ' + str(value['time']) + ' seconds\n')
                else:
                    report_file.write('Time:   ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value['time'])) + '\n')
                report_file.write('\n')

                self.dump_power(report_file, value)
                self.dump_memory(report_file, value)
                self.dump_cpu(report_file, value)
                self.dump_gpu(report_file, value)
        report_file.close()

        # Writing the raw data for final deltas
        with open(raw_data_path, 'w', encoding='utf-8') as raw_file:
            raw_file.write(f"Time_delta: {final_stats['time']}\n")
            raw_file.write(f"Power_voltage_mV_delta: {final_stats['power']['voltage_mV']}\n")
            raw_file.write(f"Power_current_mA_delta: {final_stats['power']['current_mA']}\n")
            raw_file.write(f"Power_avg_power_mW_delta: {final_stats['power']['avg_power_mW']}\n")
            
            raw_file.write(f"Memory_used_KB_delta: {final_stats['memory']['used_KB']}\n")
            raw_file.write(f"Memory_total_KB_delta: {final_stats['memory']['total_KB']}\n")
            raw_file.write(f"Memory_free_KB_delta: {final_stats['memory']['free_KB']}\n")
            raw_file.write(f"Memory_cached_KB_delta: {final_stats['memory']['cached_KB']}\n")
            raw_file.write(f"Memory_shared_KB_delta: {final_stats['memory']['shared_KB']}\n")
            
            raw_file.write(f"GPU_load_delta: {final_stats['gpu']['load']}\n")
            raw_file.write(f"GPU_curr_freq_delta: {final_stats['gpu']['curr_freq']}\n")

            for cpu in final_stats['cpu']['cpus']:
                raw_file.write(f"CPU{cpu['cpu']}_current_freq_delta: {cpu['current_freq']}\n")
        raw_file.close()