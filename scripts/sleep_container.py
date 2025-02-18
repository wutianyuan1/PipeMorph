import time
import os

if __name__ == '__main__':
    # pkgs = ['jemalloc', 'redis']
    # print("Installing: ", pkgs)
    # for pkg in pkgs:
    #     os.system(f"yum install -y {pkg}")
    sleep_time = 3600 * 12
    print(f'Sleeping for {sleep_time}')
    time.sleep(sleep_time)
