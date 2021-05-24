import numpy as np
import tensorflow as tf
from sklearn import preprocessing

basic_land_names = set(['plains','island','swamp','mountain','forest'])


def get_data_format(pkl_data):
    if isinstance(pkl_data,np.ndarray):
        return 'draft'
    elif isinstance(pkl_data,dict):
        if pkl_data['data_format']:
            return 'sparse'
    raise ValueError('Unknown pkl_data format')

def get_processor(pkl_data,**kwargs):
    data_format = get_data_format(pkl_data)
    
    if data_format == 'draft':
        return DraftFormatProcessor(pkl_data,**kwargs)
    elif data_format == 'sparse':
        return SparseFormatProcessor(pkl_data,**kwargs)
    raise ValueError('Unknown pkl_data format')


class BaseDataProcessor:
    def __init__(self,**kwargs):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def get_iter(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_set_size(self):
        return self.num_cards_in_set


class SparseFormatProcessor(BaseDataProcessor):
    def __init__(self, data_dict,**kwargs):
        self.x = data_dict['x']
        self.y = data_dict['y']
        self.len = self.y.shape[0]
        self.num_cards_in_set = self.y.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_iter(self,shuffle=True):
        
        indices = np.arange(self.__len__())
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                yield self.__getitem__(idx)
    
    def __len__(self):
        return self.len


class DraftFormatProcessor(BaseDataProcessor):
    
    def __init__(self, drafts_tensor, num_cards_in_set=None,**kwargs):
        """Initialization.
        """
        self.drafts_tensor = drafts_tensor
        if num_cards_in_set is None:
            self.num_cards_in_set = np.max(drafts_tensor) + 1
            print('inferring the number of cards in set: {}'.format(self.num_cards_in_set))
        # self.pack_size = int(self.drafts_tensor.shape[1]/3)
        # self.draft_size = self.pack_size*3
        self.pack_size = self.drafts_tensor.shape[2]
        self.draft_size = self.drafts_tensor.shape[1]
        
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
    
    def get_iter(self,shuffle=True):
        
        indices = np.arange(self.__len__())
        while True:
            if shuffle:
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
        x = np.zeros([self.num_cards_in_set * 2], dtype = "int16")
        
        #Fill in collection vector excluding current pick (first half).
        for n in self.drafts_tensor[draft_num, :pick_num, 0]:
            x[n] += 1
            
        #Fill in pack vector.
        cards_in_pack =  self.pack_size - pick_num%self.pack_size #Cards in current pack.
        for n in self.drafts_tensor[draft_num, pick_num, :cards_in_pack]:
            x[n + self.num_cards_in_set] = 1
            
        return x
    
    def create_new_y(self, pick_num, draft_num, not_in_pack=0.5):
        """Generate y, a target pick row vector.
        Picked card is assigned a value of 1.
        Other cards are assigned a value of 0.
        """
        #Initialize target vector.
        #y = np.array([0] * self.num_cards_in_set)
        y = np.zeros([self.num_cards_in_set], dtype = "int16")
            
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

