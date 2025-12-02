import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategy_a import train_strategy_a


def main(args):
    
    return train_strategy_a(args)


if __name__ == '__main__':
    print("Please run main.py in the root directory to start training")