#!/usr/bin/python
import os
import sys


def print_fw_size(fw, flash_size, ram_size):
    flash_used = 0
    ram_used = 0

    print('Memory usage:')
    print(f'FLASH: {flash_used}/{flash_size}B')
    print(f'RAM:   {ram_used}/{ram_size}B')
    pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise TypeError(f'function takes 3 arguments ({len(sys.argv)} given)')

    fw = sys.argv[1]
    flash_size = sys.argv[2]
    ram_size = sys.argv[3]
    print_fw_size(fw, flash_size, ram_size)
