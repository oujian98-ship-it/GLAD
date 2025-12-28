import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from gald_dc import train_gald_dc


def main(args):
    return train_gald_dc(args)


if __name__ == '__main__':
    print("Please run main.py in the root directory to start training")