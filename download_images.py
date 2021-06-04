
import pandas as pd
import argparse
from set_utils import set_metadata_map
import os 
from os.path import join as pjoin
import requests
script_path = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--force-download", action="store_true", dest="force",
        help=("Whether to force download of images (by default will skip already downloaded images) ")
    )
    parser.add_argument(
        "--set_codes", action="store", dest="set_codes",nargs='+',
        default=None,
        help=("List of set_codes to download ('stx'/'m19'), by default downloads all available sets")
    )

    args = parser.parse_args()
    force_download = args.force
    set_codes = args.set_codes
    set_codes = set([x.lower() for x in set_codes])
    image_root = pjoin(script_path,'images')
    os.makedirs(image_root,exist_ok=True)

    
    for set,metadata in set_metadata_map.items():
        if set_codes is not None and set.lower() not in set_codes:
            continue
        print('[{}]Downloading images...'.format(set)) 
        draft_creator = metadata.load_draft_creator()
        draft_creator.download_images(force_download=args.force,
                                        verbose=1)
            

