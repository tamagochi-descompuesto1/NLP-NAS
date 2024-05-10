from jtop import jtop


if __name__ == '__main__':
    with jtop() as jetson:
        count = 0
        
        print('---------    jetson.stats    --------')
        while jetson.ok() and count < 5:
            stats = jetson.stats

            print('*CPU1:        ', stats['CPU1'])
            print('*CPU2:        ', stats['CPU2'])
            print('*CPU3:        ', stats['CPU3'])
            print('*CPU4:        ', stats['CPU4'])
            print('*CPU5:        ', stats['CPU5'])
            print('*CPU6:        ', stats['CPU6'])
            print('*RAM:         ', stats['RAM'])
            print('*GPU:         ', stats['GPU'])
            print('*Power TOT:   ', stats['Power TOT'])
            print('*nvp model:   ', stats['nvp model'])
            print()

            count +=1
        
        count = 0
        print('-------- jetson.cpu  --------')
        while jetson.ok() and count < 5:
            cpu_stats = jetson.cpu

            print('TOTAL CPU STATS:')
            print(' *User:     ', cpu_stats['total']['user'])
            print(' *Nice:     ', cpu_stats['total']['nice'])
            print(' *System:   ', cpu_stats['total']['system'])
            print(' *Idle:     ', cpu_stats['total']['idle'])

            for index, cpu in enumerate(cpu_stats['cpu']):
                print('CPU ', index + 1, ' STATS:')
                print(' *Curr Freq:     ', cpu['freq']['cur'], 'kHz')
                print(' *User:          ', cpu['user'])
                print(' *Nice:          ', cpu['nice'])
                print(' *System:        ', cpu['system'])
                print(' *Idle:          ', cpu['idle'])
            print()

            count +=1

        count = 0
        print('-------- jetson.memory  --------')
        while jetson.ok() and count < 5:
            memory_stats = jetson.memory

            print('RAM STATS:')
            print(' *Used RAM:      ', memory_stats['RAM']['used'], 'KB/', memory_stats['RAM']['tot'], 'KB')
            print(' *Free:          ', memory_stats['RAM']['free'], 'KB')
            print(' *Chached:       ', memory_stats['RAM']['cached'], 'KB')
            print(' *Shared:        ', memory_stats['RAM']['shared'], 'KB')
            print('SWAP STATS:')
            print(' *Used SWAP:       ', memory_stats['SWAP']['used'], 'KB/', memory_stats['SWAP']['tot'], 'KB')
            print(' *Cached:          ', memory_stats['SWAP']['cached'], 'KB')
            print('EMC STATS:')
            print(' *Min Freq:       ', memory_stats['EMC']['min'], 'kHz')
            print(' *Max Freq:       ', memory_stats['EMC']['max'], 'kHz')
            print(' *Curr Freq:      ', memory_stats['EMC']['cur'], 'kHz')
            print()

            count +=1

        count = 0
        print('-------- jetson.gpu  --------')
        while jetson.ok() and count < 5:
            gpu_stats = jetson.gpu

            print('GPU ga10b STATS:')
            print(' *Type:              ', gpu_stats['ga10b']['type'])
            print(' *Load:              ', gpu_stats['ga10b']['status']['load'])
            print(' *Min Freq:          ', gpu_stats['ga10b']['freq']['min'])
            print(' *Max Freq:          ', gpu_stats['ga10b']['freq']['max'])
            print(' *Curr Freq:         ', gpu_stats['ga10b']['freq']['cur'])
            print(' *Power Control:     ', gpu_stats['ga10b']['power_control'])
            print()

            count +=1

        count = 0
        print('-------- jetson.power  --------')
        while jetson.ok() and count < 5:
            power_stats = jetson.power

            print('TOTAL POWER STATS:')
            print(' *Voltage:                                ', power_stats['tot']['volt'], 'mV')
            print(' *Current:                                ', power_stats['tot']['curr'], 'mA')
            print(' *Avg Current Limit(warn):                ', power_stats['tot']['warn'], 'mA')
            print(' *Instantanous Current Limit(crit):       ', power_stats['tot']['crit'], 'mA')
            print(' *Power:                                  ', power_stats['tot']['power'], 'mW')
            print(' *Avg Power:                              ', power_stats['tot']['avg'], 'mW')
            print('VDD_CPU_GPU_CV POWER STATS:')
            print(' *Voltage:                                ', power_stats['rail']['VDD_CPU_GPU_CV']['volt'], 'mV')
            print(' *Current:                                ', power_stats['rail']['VDD_CPU_GPU_CV']['curr'], 'mA')
            print(' *Avg Current Limit(warn):                ', power_stats['rail']['VDD_CPU_GPU_CV']['warn'], 'mA')
            print(' *Instantanous Current Limit(crit):       ', power_stats['rail']['VDD_CPU_GPU_CV']['crit'], 'mA')
            print(' *Power:                                  ', power_stats['rail']['VDD_CPU_GPU_CV']['power'], 'mW')
            print(' *Avg Power:                              ', power_stats['rail']['VDD_CPU_GPU_CV']['avg'], 'mW')
            print('VDD_SOC POWER STATS:')
            print(' *Voltage:                                ', power_stats['rail']['VDD_SOC']['volt'], 'mV')
            print(' *Current:                                ', power_stats['rail']['VDD_SOC']['curr'], 'mA')
            print(' *Avg Current Limit(warn):                ', power_stats['rail']['VDD_SOC']['warn'], 'mA')
            print(' *Instantanous Current Limit(crit):       ', power_stats['rail']['VDD_SOC']['crit'], 'mA')
            print(' *Power:                                  ', power_stats['rail']['VDD_SOC']['power'], 'mW')
            print(' *Avg Power:                              ', power_stats['rail']['VDD_SOC']['avg'], 'mW')
            print()

            count +=1

        count = 0
        print('-------- jetson.nvpmodel  --------')
        while jetson.ok() and count < 5:
            print(jetson.nvpmodel)
            print()
            count +=1