import time
from jtop import jtop

class JtopStats:
    def __init__(self)-> None:
        self.power = {}
        self.memory = {}
        self.cpu = {}
        self.gpu = {}
        self.stats = {}

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
        stats['time'] = time.time()
        return stats

    def __get_stats(self) -> dict:
        with jtop() as jetson:
            power = {
                'voltage_mV': jetson.power['tot']['volt'],
		        'current_mA': jetson.power['tot']['curr'],
		        'avg_power_mW': jetson.power['tot']['avg']  
            }

            memory = {
                'used_KB': jetson.memory['RAM']['used'],
                'total_KB': jetson.memory['RAM']['tot'],
                'free_KB': jetson.memory['RAM']['free'],
                'cached_KB': jetson.memory['RAM']['cached'],
                'shared_KB': jetson.memory['RAM']['shared']
            }

            cpu = {
                'total': {
                    'user': jetson.cpu['total']['user'],
                    'nice': jetson.cpu['total']['nice'],
                    'system': jetson.cpu['total']['system'],
                    'idle': jetson.cpu['total']['idle'],
                },
                'cpus': [{'cpu': index + 1, 'current_freq': cpu['freq']['cur']} for index, cpu in enumerate(jetson.cpu['cpu'])]
            }

            gpu = {
                'load': jetson.gpu['ga10b']['status']['load'],
                'min_freq': jetson.gpu['ga10b']['freq']['min'],
                'max_freq': jetson.gpu['ga10b']['freq']['max'],
                'curr_freq': jetson.gpu['ga10b']['freq']['cur']
            }

            stats = {
                'power': power,
                'memory': memory,
                'cpu': cpu,
                'gpu': gpu
            }

            stats = self.__init_time(stats)
            return stats

    def set_stats(self) -> None:
        self.stats = self.__get_stats()
        self.power = self.stats['power']
        self.memory = self.stats['memory']
        self.cpu = self.stats['cpu']
        self.gpu = self.stats['gpu']


    def __get_deltas(self) -> dict:
        if self.stats == {}:
            self.set_stats()

        current_stats = self.__get_stats()
        deltas = {}

        # Deltas for everything excep cpu and time
        for key in ['power', 'memory', 'gpu']:
            deltas[key] = {stat_value: current_stats[key][stat_value] - self.__dict__[key][stat_value] for stat_value in current_stats[key].keys()}

        # Deltas for cpu
        deltas['cpu'] = {
            'total': {
                'user': current_stats['cpu']['total']['user'] - self.__dict__['cpu']['total']['user'], 
                'nice': current_stats['cpu']['total']['nice'] - self.__dict__['cpu']['total']['nice'],
                'system': current_stats['cpu']['total']['system'] - self.__dict__['cpu']['total']['system'],
                'idle': current_stats['cpu']['total']['idle'] - self.__dict__['cpu']['total']['idle']
            },
            'cpus': [{'cpu': index + 1, 'current_freq': current_stats['cpu']['cpus'][index]['current_freq'] - cpu['current_freq']} for index, cpu in enumerate(self.__dict__['cpu']['cpus'])]
        }

        # Delta for time
        deltas['time'] = current_stats['time'] - self.__dict__['stats']['time']

        return deltas, current_stats
    
    def dump_deltas(self, path:str='/') -> None: 
        deltas, current_stats = self.__get_deltas()

        with open(path, 'w', encoding='utf-8') as file: 
            for title, value in {'STARTING STATS': self.stats, 'ENDING STATS': current_stats, 'CONSUMPTION STATS': deltas}.items():
                file.write('----------------' + title + '----------------' + '\n')
                if title == 'CONSUMPTION STATS':
                    file.write('Time:   ' + str(value['time']) + ' seconds\n')
                else:
                    file.write('Time:   ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value['time'])) + '\n')
                file.write('\n')

                file.write('POWER:\n')
                file.write('    - Voltage:        ' +  str(value['power']['voltage_mV']) + ' mV\n')
                file.write('    - Current:        ' +  str(value['power']['current_mA']) + ' mA\n')
                file.write('    - Average power:  ' +  str(value['power']['avg_power_mW']) + ' mW\n')
                file.write('\n')

                file.write('MEMORY:\n')
                file.write('    - Used RAM:          ' +  str(value['memory']['used_KB']) + ' KB\n')
                file.write('    - Total RAM:         ' +  str(value['memory']['total_KB']) + ' KB\n')
                file.write('    - Free RAM:          ' +  str(value['memory']['free_KB']) + ' KB\n')
                file.write('    - Cached RAM:        ' +  str(value['memory']['cached_KB']) + ' KB\n')
                file.write('    - Shared RAM:        ' + str(value['memory']['shared_KB']) + ' KB\n')
                file.write('\n')

                file.write('CPU:\n')
                file.write('    - Total CPU stats:\n')
                file.write('        + User utilization:   ' + str(value['cpu']['total']['user']) + '\n')
                file.write('        + Nice utilization:   ' + str(value['cpu']['total']['nice']) + '\n')
                file.write('        + System utilization: ' + str(value['cpu']['total']['system']) + '\n')
                file.write('        + Idle utilization:   ' + str(value['cpu']['total']['idle']) + '\n')
                for cpu in value['cpu']['cpus']:
                    file.write('    - CPU ' + str(cpu['cpu']) + ' frequency:   ' + str(cpu['current_freq']) + ' kHz\n')
                file.write('\n')

                file.write('GPU:\n')
                file.write('    - GPU load:                  ' +  str(value['gpu']['load']) + '\n')
                file.write('    - Minimum frequency:         ' +  str(value['gpu']['min_freq']) + ' kHz\n')
                file.write('    - Maximum frequency:         ' + str(value['gpu']['max_freq']) + ' kHz\n')
                file.write('    - Current frequency:         ' +  str(value['gpu']['curr_freq']) + ' kHz\n')
                file.write('\n')

        file.close()