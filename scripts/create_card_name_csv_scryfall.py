import sys
import os

parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(parent_dir_name)
sys.path.append(parent_dir_name)
from providers import scryfall
import argparse


def main(args):
    scryfall.create_set_csv_from_scryfall_oracle_json(
                                                args.oracle_json,
                                                args.set_codes,
                                                args.output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--card_json", action="store", dest="oracle_json",
        required=True,
        help=""
    )
    parser.add_argument(
        "--output_csv", action="store", dest="output_csv",
        required=True,
        help=""
    )
    parser.add_argument(
        "--set_codes", action="store", dest="set_codes",
        required=True,nargs='+',
        help="all set codes to filter for ('stx' and 'sta' for strixhaven and strixhaven mystical archive)"
    )
    args = parser.parse_args()
    main(args)