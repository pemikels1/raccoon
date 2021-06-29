import argparse
import os


# create argument parser, add hyperparameters, and parse them
def park_parse_args():

    parser = argparse.ArgumentParser(description='Raccoon Detection')
    parser.add_argument('--batch_size', default=4, help='Number of observations sent at a time')
    parser.add_argument('--epochs', default=10, help='Number of passes through dataset')
    parser.add_argument('--shuffle', default=True, help='Shuffle data after every epoch')
    parser.add_argument('--lr', default=0.005, help='Initial learning rate')
    parser.add_argument('--train_percentage', default=0.6, help='Percentage of data in training set')
    parser.add_argument('--seed', default=95, help='Random seed')
    parser.add_argument('--data', default=os.environ['DATA_PATH'], help='Path to data folder')
    parser.add_argument('--checkpoint', default=os.environ['CHECKPOINTS_PATH'], help='Path to checkpoints folder')
    parser.add_argument('--torch_files', default=os.environ['TORCH_FILES_PATH'], help='Path to torch_files folder')

    args = parser.parse_args()

    return args
