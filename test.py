import argparse
import sys
from fvcore.common.config import CfgNode

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='train', type=str, help='[train]')
    parser.add_argument(
        "--opts",
        help="",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    print(parse_args())