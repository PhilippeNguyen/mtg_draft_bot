import numpy as np
import datetime
import pandas as pd
from operator import itemgetter
from copy import deepcopy

class BotTester(object):
    """The BotTester object is used to evaluate how close a collection of bot's picks match human picks.
    
    This can be used in the following manner:
        tester = BotTester(drafts)
        tester.evaluate_bots([bot], ["SGD"])
        tester.write_rating_dict()
    """
   
    def __init__(self, drafts):
        """Create a new BotTester instance.

        Fields:
          self.drafts - a collection of multiple draft objects (list of list of list of cardnames)
          self.correct - DataFrame of all bots' correct choices compared to human picks
          self.fuzzy_correct - DataFrame of all bots' correct choices (if human pick in top 3 bot picks)
          self.card_acc - DataFrame of per-card accuracy metrics for all bots
        
        :param drafts: Attach a set of drafts to the BotTester
        """
        before = datetime.datetime.now()
        self.drafts = drafts
        self.n_packs = len(drafts)*45
        self.correct = pd.DataFrame(columns = ['draft_num', 'pick_num', 'human_pick'], index = range(self.n_packs))
        self.fuzzy_correct = pd.DataFrame(columns = ['draft_num', 'pick_num', 'human_pick'], index = range(self.n_packs))
        self.rank_error = pd.DataFrame(columns = ['draft_num', 'pick_num', 'human_pick'], index = range(self.n_packs))
        self.card_acc = pd.DataFrame(columns = ['human_pick'])
        print("Initialization time taken: " + str(datetime.datetime.now() - before))

    def evaluate_bots(self, bots, bot_names):
        """Evaluates accuracy and fuzzy accuracy of a list of bots. 
        
        "Correct" is whether or not the bot's top choice matched the human's top choice.
        "Fuzzy correct" is whether or not the human's top choice was in the bot's top 3 choices.
        These values are stored the DataFrames acc and fuzz_acc for all bots.

        :param bots: List of bots that all inherit from "bot.py"
        :param bot_names: List of bot names (strings) of the same size as the list of bots.
        """

        # Checks if we need to initialize dataframes
        initialize = np.isnan(self.correct.iloc[0,2])

        # Builds up static values as lists to later add to dataframes
        draft_num_list = [None]*self.n_packs
        pick_num_list = [None]*self.n_packs
        human_pick_list = [None]*self.n_packs

        # Fills in dataframes of correct choices
        temp_names = []
        before = datetime.datetime.now()
        static_cols = ['draft_num', 'pick_num', 'human_pick']
        for bot_counter in range(len(bots)): # AKh: better to rename to iBot
            bot = bots[bot_counter]
            all_correct = [None]*self.n_packs
            all_fuzzy = all_correct.copy()
            all_rank_error = all_correct.copy()
            pack_counter = 0
            for draft_num in range(len(self.drafts)):
                draft = self.drafts[draft_num]
                collection = []
                for pick_num in range(len(draft)):
                    pack = draft[pick_num]

                    # Stores draft and pick number in dataframes if uninitialized
                    if initialize: 
                        draft_num_list[pack_counter] = draft_num + 1
                        pick_num_list[pack_counter] = pick_num + 1
                        human_pick_list[pack_counter] = pack[0]

                    # Gets bot ranking on the current pack
                    pack_rank = bot.rank_pack([pack, collection])
                    collection.append(bot.get_top_pick(pack_rank))

                    # Gets top-one and top-three accuracy for the current pack
                    exact_correct = self.is_bot_correct(pack, pack_rank)
                    fuzzy_correct = self.is_bot_correct(pack, pack_rank, fuzzy = True)
                    rank_error = self.get_rank_error(pack, pack_rank)

                    # Stores accuracy in dataframes
                    all_correct[pack_counter] = exact_correct[1]
                    all_fuzzy[pack_counter] = fuzzy_correct[1]
                    all_rank_error[pack_counter] = rank_error[1]
                    pack_counter += 1

            # Only initializes dataframe values once
            if initialize:
                self.correct['draft_num'] = draft_num_list
                self.correct['pick_num'] = pick_num_list
                self.correct['human_pick'] = human_pick_list
                self.fuzzy_correct['draft_num'] = draft_num_list
                self.fuzzy_correct['pick_num'] = pick_num_list
                self.fuzzy_correct['human_pick'] = human_pick_list
                self.rank_error['draft_num'] = draft_num_list
                self.rank_error['pick_num'] = pick_num_list
                self.rank_error['human_pick'] = human_pick_list
                initialize = False

            # Stores accuracy info in a single column of existing dataframes
            bot_name = bot_names[bot_counter]
            self.correct[bot_name] = all_correct
            self.fuzzy_correct[bot_name] = all_fuzzy
            self.rank_error[bot_name] = all_rank_error
            current = datetime.datetime.now() 
            print(bot_name + " time taken: " + str(current - before))
            before = current

        # Fills in dataframes of per-card accuracies
        unique_cards = np.sort(self.correct['human_pick'].unique())
        self.card_acc['human_pick'] = unique_cards # All card names; human_pick is just where they came from
        for bot_name in bot_names:
            accuracies = []
            for human_pick in unique_cards:
                all_picks = self.correct.loc[self.correct['human_pick'] == human_pick]
                accuracies.append(all_picks[bot_name].sum() / all_picks.shape[0])
            self.card_acc[bot_name] = accuracies

    def write_evaluations(self, exact_filename = "output_files/exact_correct.tsv", fuzzy_filename = "output_files/fuzzy_correct.tsv", 
                          rank_error_filename = "output_files/rank_error.tsv", acc_filename = "output_files/card_accuracies.tsv"):
        """Writes correctness and accuracy DataFrames to filenames.
        """
        self.correct.to_csv(exact_filename, sep = "\t", index = False)
        print("Wrote correct to: " + str(exact_filename))
        self.fuzzy_correct.to_csv(fuzzy_filename, sep = "\t", index = False)
        print("Wrote fuzzy_correct to: " + str(fuzzy_filename))
        self.rank_error.to_csv(rank_error_filename, sep = "\t", index = False)
        print("Wrote rank_error to: " + str(rank_error_filename))
        self.card_acc.to_csv(acc_filename, sep = "\t", index = False)
        print("Wrote card_acc to: " + str(acc_filename))
    
    def report_evaluations(self):
        '''Reports some minimal info on bot running results in the notebook,
        good for quick troubleshooting'''
        
        print(np.mean(self.correct))
    
    def is_bot_correct(self, pack, pack_rank, fuzzy = False):
        """ Checks whether or not a bot's pick matches a human's pick.
        
        Returns a tuple of (cardname, bot_correct) for whether or not the
        bot's top choice matched the human's choice. If fuzzy = True, then
        instead the bot is correct if the human's choice is in bot's top 3
        """
        bot_correct = 0
        human_pick = pack[0]
        pack_rank = sorted(pack_rank, key = pack_rank.get, reverse = True)
        if not fuzzy:
            bot_pick = pack_rank[0]
            if human_pick == bot_pick:
                bot_correct = 1
        elif fuzzy:
            for i in range(min(len(pack_rank), 3)):
                bot_pick = pack_rank[i]
                if human_pick == bot_pick:
                    bot_correct = 1
        return (human_pick, bot_correct)
    
    def get_rank_error(self, pack, pack_rank):
        """ Checks the rank error between a bot pick and a human pick.
        
        Returns a tuple of (cardname, rank_error) for the rank of the human's choice. 
        """
        rank_error = 0
        human_pick = pack[0]
        pack_rank = sorted(pack_rank, key = pack_rank.get, reverse = True)
        for card in pack_rank:
            if card == human_pick:
                break
            rank_error += 1
        return (pack[0], rank_error)

