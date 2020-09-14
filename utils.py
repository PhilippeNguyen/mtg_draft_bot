import numpy as np
import tensorflow as tf
from sklearn import preprocessing

basic_land_names = set(['Plains','Island','Swamp','Mountain','Forest'])

class DataProcessor(object):
    
    def __init__(self, drafts_tensor, le):
        """Initialization.
        """
        self.drafts_tensor = drafts_tensor
        self.le = le
        self.cards_in_set = len(self.le.classes_)
        self.pack_size = int(self.drafts_tensor.shape[1]/3)
        self.draft_size = self.pack_size*3
        
    def __getitem__(self, index):
        """Return a training example.
        """
        #Grab information on current draft.
        pick_num = index % self.draft_size #0-self.pack_size*3-1
        draft_num = int((index - pick_num)/self.draft_size)
        
        #Generate.
        x = self.create_new_x(pick_num, draft_num)
        y = self.create_new_y(pick_num, draft_num)
        return x, y
    
    def get_iter(self):
        
        indices = np.arange(self.__len__())
        np.random.shuffle(indices)
        
        for idx in indices:
            yield self.__getitem__(idx)
    
    def create_new_x(self, pick_num, draft_num):
        """Generate x, input, as a row vector.
        0:n     : collection vector
                  x[i]=n -> collection has n copies of card i
        n:2n    : pack vector
                  0 -> card not in pack
                  1 -> card in pack
        Efficiency optimization possible. Iterative adds to numpy array.
        """
        #Initialize collection / cards in pack vector.
        x = np.zeros([self.cards_in_set * 2], dtype = "int16")
        
        #Fill in collection vector excluding current pick (first half).
        for n in self.drafts_tensor[draft_num, :pick_num, 0]:
            x[n] += 1
            
        #Fill in pack vector.
        cards_in_pack =  self.pack_size - pick_num%self.pack_size #Cards in current pack.
        for n in self.drafts_tensor[draft_num, pick_num, :cards_in_pack]:
            x[n + self.cards_in_set] = 1
            
        return x
    
    def create_new_y(self, pick_num, draft_num, not_in_pack=0.5):
        """Generate y, a target pick row vector.
        Picked card is assigned a value of 1.
        Other cards are assigned a value of 0.
        """
        #Initialize target vector.
        #y = np.array([0] * self.cards_in_set)
        y = np.zeros([self.cards_in_set], dtype = "int16")
            
        #Add picked card.
        y[self.drafts_tensor[draft_num, pick_num, 0]] = 1
        return y



    def __len__(self):
        return len(self.drafts_tensor) * self.draft_size


def create_set_vector(casting_cost, card_type, rarity, color_vector):
    """
    Returns a feature-encoded card property vector. 
    
    There are 21 binary features:
    
    0. cmc=0
    1. cmc=1
    2. cmc=2
    3. cmc=3
    4. cmc=4
    5. cmc=5
    6. cmc=6
    7. cmc>=7
    8. creature?
    9. common?
    10. uncommon?
    11. rare?
    12. mythic?
    13. colorless?
    14. monocolored?
    15. multicolored?
    16. color1?
    17. color2?
    18. color3?
    19. color4?
    20. color5?
    
    :param casting_cost: integer casting cost of card
    :param card_type: "Creature" or other
    :param rarity": "C", "U", "R", or "M"
    "param color_vector": vector corresponding to colors of card, example: [1,0,0,0,1]
    
    """
    # Initialize set vector
    v = [0] * 21
    
    # Encode cmc
    if casting_cost == 0:
        v[0] = 1
    elif casting_cost == 1:
        v[1] = 1
    elif casting_cost == 2:
        v[2] = 1
    elif casting_cost == 3:
        v[3] = 1
    elif casting_cost == 4:
        v[4] = 1
    elif casting_cost == 5:
        v[5] = 1
    elif casting_cost == 6:
        v[6] = 1
    elif casting_cost >= 7:
        v[7] = 1
    else:
        print("WARNING: Undefined casting cost.")
    
    # Encode type
    if card_type == "Creature":
        v[8] = 1
        
    # Encode rarity
    if rarity == "C":
        v[9] = 1
    elif rarity == "U":
        v[10] = 1
    elif rarity == "R":
        v[11] = 1
    elif rarity == "M":
        v[12] = 1
    
    # Process number of colors
    num_colors = len([c for c in color_vector if c > 0])
    if num_colors == 0:
        v[13] = 1
    elif num_colors == 1:
        v[14] = 1
    elif num_colors >= 2:
        v[15] = 1
    
    # Process card color
    if color_vector[0] > 0:
        v[16] = 1
    if color_vector[1] > 0:
        v[17] = 1
    if color_vector[2] > 0:
        v[18] = 1
    if color_vector[3] > 0:
        v[19] = 1
    if color_vector[4] > 0:
        v[20] = 1
    return v

def cmc_from_string(cmc_string):
    """
    Return an integer converted mana cost from cmc_string. 
    
    Each character adds 1 to cmc. 
    
    :param cmc_string: String or integer representation of cmc. Example: "1UBR".
    :returns: Integer cmc. Example: 4.
    """
    # If int, we are done
    if type(cmc_string) is int:
        return cmc_string
    
    # Convert string to integer cmc
    cmc = 0
    digit_string = ""
    letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    digits = set("1234567890")
        
    for c in cmc_string:        
        if c in letters:
            cmc += 1
        else:
            digit_string += c
    if len(digit_string) > 0:
        cmc += int(digit_string)
    return cmc




def create_set_tensor(magic_set):
    """
    Returns a set tensor which represents the properties of cards in the set.
    
    There are M features and N cards in the set and the tensor is of size M x N.
    
    The features are documented in the create_set_vector() function. 
    """
    set_list = []
    
    # Requires these names to be present in the set file 
    reduced_set = magic_set[["Name", "Casting Cost 1", "Card Type", "Rarity", "Color Vector"]]
    for index, row in reduced_set.iterrows():
        card_vector = create_set_vector(cmc_from_string(row[1]), row[2], row[3], row[4])
        set_list.append(card_vector)

    set_tensor = np.asarray(set_list)
    return set_tensor


class NNBot(object):
    pass
class DraftCreator(object):

    def __init__(self,set_df,land_df):
        self.set_df = set_df
        self.land_df = land_df

        self.mythic_prob = 1/8
        self.num_common = 10
        self.num_uncommon = 3
        self.num_rare = 1
        self.num_land = 1

        self.build_set()

    def build_set(self):
        self.commons = []
        self.uncommons = []
        self.rares = []
        self.mythic_rares = []
        self.rarities_all_data = {'C': self.commons,
                    'U': self.uncommons,
                    'R': self.rares,
                    'M': self.mythic_rares,
                    }
        self.rarities_idx = {'C': [],
                    'U': [],
                    'R': [],
                    'M': [],
                    }
        self.name_dict = {}
        for row_idx,row in self.set_df.iterrows():
            self.rarities_all_data[row['Rarity']].append((row_idx,row))
            self.rarities_idx[row['Rarity']].append(row_idx)
            self.name_dict[row['Name']]  = row
        for key,val in self.rarities_idx.items():
            self.rarities_idx[key] = np.asarray(val)

        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(self.set_df["Name"].values)

        self.land_names = basic_land_names.copy()
        for row_idx,row in self.land_df.iterrows():
            land_name = row['Name']
            if land_name.split('_')[0] not in basic_land_names:
                self.land_names.add(land_name)
                
        self.land_idxs = self.label_encoder.transform(list(self.land_names))


    def create_pack(self):
        pack = []
        
        for _ in range(self.num_rare):
            if np.random.uniform() < self.mythic_prob:
                pack.extend(np.random.choice(self.rarities_idx['M'],size=1,replace=False))
            else:
                pack.extend(np.random.choice(self.rarities_idx['R'], size=1, replace=False))
        pack.extend(np.random.choice(self.rarities_idx['U'], size=self.num_uncommon, replace=False))
        pack.extend(np.random.choice(self.rarities_idx['C'], size=self.num_common, replace=False))
        pack.extend(np.random.choice(self.land_idxs, size=1, replace=False))

        return np.asarray(pack)


    def read_pack(self,pack):
        '''

        '''
        if len(pack.shape) != 1:
            raise Exception('pack must be 1 dimensional')
        pack_names = self.label_encoder.inverse_transform(pack)
        for pack_name in pack_names:
            print(self.name_dict[pack_name])

    def card_info_from_idx(self,card_idx):
        return self.name_dict[self.label_encoder.inverse_transform([card_idx])[0]]

    def card_name_from_idx(self,card_idx):
        return self.label_encoder.inverse_transform([card_idx])[0]

    def card_and_cc_from_idx(self,card_idx):
        card_info = self.name_dict[self.label_encoder.inverse_transform([card_idx])[0]]
        return
    def read_bot_picks(self,bot_pick_vec,verbosity=2):
        '''
            bot_pick_vec should be of shape (set_size),
            with the value of each idx
        '''

        non_zeros = np.nonzero(bot_pick_vec)[0]
        deck = []
        for card_idx in non_zeros:
            num_card = str(int(bot_pick_vec[card_idx]))
            card_info = self.card_info_from_idx(card_idx)
            cast_cost = card_info['Casting Cost 1']
            cmc = cmc_from_string(cast_cost)

            card_str = ''
            if verbosity >= 0:
                card_str = card_info['Name'] +card_str
            if verbosity >= 1:
                card_str = '{:10}'.format(cast_cost) +card_str
            if verbosity >= 2:
                card_str = '{:3}'.format(card_info['Rarity']) + card_str
            card_str = '{:3}'.format(num_card) + card_str
            deck.append((cmc,card_str))

        deck = sorted(deck,key=lambda x:x[0])

        for cmc,card_str in deck:
            print(card_str)