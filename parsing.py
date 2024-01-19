import argparse

parser = argparse.ArgumentParser(description='MY_SR_model')

parser.add_argument('--SR_factor', type=int, default='6', help = 'SR factor should be integer')

parser.add_argument('--dir_path', type=str, default='./COLON/TRAIN_SLICES/HR2', help = 'dir path')

parser.add_argument('--batch', type=int, default='32', help = 'Batch size')

parser.add_argument('--loss', type=str, default='L1', help = 'Loss Function')

parser.add_argument('--loss', type=str, default='L1', help = 'Loss Function')


args = parser.parse_args()