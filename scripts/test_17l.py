import sys
import os

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(parent_dir_name)
sys.path.append(parent_dir_name)
from providers import seventeen_lands
import argparse


def main(args):
    seventeen_lands.create_card_name_csv_from_draft_csv(
                                                args.draft_csv,
                                                args.output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--draft_csv", action="store", dest="draft_csv",
        required=True,
        help=""
    )
    parser.add_argument(
        "--output_csv", action="store", dest="output_csv",
        required=True,
        help=""
    )
    args = parser.parse_args()
    main(args)