import argparse

from referit_loader import ReferDataset

parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='datasets/refer',
                    help='path to ReferIt splits data folder')
parser.add_argument('--dataset', default='unc', type=str,
                    help='dataset used to train network')
parser.add_argument('--split-root', type=str, default='data',
                    help='path to dataloader splits data folder')
parser.add_argument('--parser-url', type=str, default='http://localhost:9000',
                    help='url of stanford corenlp server')
args = parser.parse_args()

if __name__ == '__main__':

    refer = ReferDataset(data_root=args.data,
                        dataset=args.dataset,
                        split_root=args.split_root,
                        parser_url=args.parser_url)