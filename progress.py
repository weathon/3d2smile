import os
import time
import sys
import psutil
from colorama import Fore, Style

total_files = 80000
directory = '/home/wg25r/rendered'

def count_files(directory):
    return len(os.listdir(directory))

def progress_bar(count, total, prefix='', suffix='', length=50, fill='█'):
    percent = round(100 * count / float(total), 1)
    filled_length = int(length * count // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    color_prefix = Fore.GREEN + prefix + Style.RESET_ALL
    color_bar = Fore.CYAN + bar + Style.RESET_ALL
    color_percent = Fore.YELLOW + str(percent) + '%' + Style.RESET_ALL

    print('\r%s |%s| %s %s' % (color_prefix, color_bar, color_percent, suffix))


def calculate_eta(start_time, count, total):
    elapsed_time = time.time() - start_time
    eta = (elapsed_time / (count - already_done + 1)) * (total - already_done - count)
    return time.strftime('%H:%M:%S', time.gmtime(eta))

def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return f'CPU: {cpu_usage}%'

already_done = len(os.listdir(directory))

def main():
    start_time = time.time()
    prefix = 'Progress:'
    
    while True:
        count = count_files(directory)
        finished = min(count, total_files)
        remaining = max(total_files - count, 0)
        eta = calculate_eta(start_time, count, total_files)
        cpu = get_cpu_usage()

        os.system("clear")

        print(f'Files: {finished}/{total_files} Finished | {remaining} Remaining')
        print(f'ETA: {eta} | {cpu}')
        progress_bar(count, total_files, prefix)
        
        if count >= total_files:
            break
        
        time.sleep(1)  # Sleep for 1 second before checking again
    
    print()  # Print a new line after the progress bar
    print('File count reached the target.')

if __name__ == '__main__':
    main()
