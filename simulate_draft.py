from nn_utils import NNBot
import argparse
import keras
import sys
from set_utils import get_set_metadata
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_hdf5", action="store", dest="model_hdf5",
        required=True,
        help=("path to the model_hdf5 ")
    )
    parser.add_argument(
        "--set_code", action="store", dest="set_code",
        required=True,
        help=("3-symbol code for the set, (only 'm19' and 'stx' available atm) ")
    )

    args = parser.parse_args()
    model = keras.models.load_model(args.model_hdf5)
    set_code = args.set_code
    set_metadata = get_set_metadata(set_code)
    draft_coord = set_metadata.load_draft_creator()

    num_bots = 8
    num_packs = 3
    num_cards_per_pack = 15

    #create bots
    bot_list = []
    for _ in range(num_bots):
        nnbot = NNBot(keras_model=model,
                      set_size=draft_coord.get_set_size(),
                      pick_mode='max'
                      )
        bot_list.append(nnbot)

    #simulate the draft
    for pack_num in range(num_packs):
        packs = []
        if pack_num %2 == 0:
            dir = 1
        else:
            dir = -1
        for _ in range(num_bots):
            packs.append(draft_coord.create_pack())

        for pick_num in range(num_cards_per_pack):
            new_packs = [[] for _ in range(num_bots)]
            for bot_idx,nnbot in enumerate(bot_list):
                sys.stdout.write('\r >> Pack : {} , Pick {} , Bot {}'.format(pack_num,pick_num,bot_idx))
                sys.stdout.flush()
                pack = packs[bot_idx]
                out_pack,_ = nnbot.pick_and_add(pack)
                out_bot_idx = (bot_idx + dir) %(num_bots)
                new_packs[out_bot_idx] = out_pack
            packs = new_packs



    for bot_idx,nnbot in enumerate(bot_list):
        print()
        print('Bot_Num: {}'.format(bot_idx))

        print(draft_coord.read_pool_picks(nnbot.picks))
