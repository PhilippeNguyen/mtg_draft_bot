import numpy as np
from sklearn import preprocessing
import os
from os.path import join as pjoin
import pandas as pd
from utils import basic_land_names
file_path = os.path.abspath(__file__)
metadata_path = pjoin(os.path.split(file_path)[0],'metadata')
images_path = pjoin(os.path.split(file_path)[0],'images')
from itertools import product
import requests




class BaseDraftCreator:
    rarity_ordering = {'c':0,'u':1,'r':2,'m':3,
                        'C':0,'U':1,'R':2,'M':3,
                        'common':0,'uncommon':1,'rare':2,'mythic':3,}
    def create_pack(self):
        raise NotImplementedError
    def read_pack(self):
        raise NotImplementedError
    def get_set_size(self):
        raise NotImplementedError
    def read_pool_picks(self):
        raise NotImplementedError

    def get_set_size(self):
        return len(self.set_df)
    
    def download_images(self,force_download=False,verbose=0):
        image_set_folder = pjoin(images_path,self.set_code)
        os.makedirs(image_set_folder,exist_ok=True)
        for row_idx,row in self.set_df.iterrows():
            if verbose:
                print('{}/{}'.format(row_idx,len(self.set_df)),end='\r')
            img_name = '{:04d}.jpg'.format(row_idx)
            img_path = pjoin(image_set_folder,img_name)
            
            if not force_download and os.path.exists(img_path):
                continue
            url = row['Image URL']
            r = requests.get(url, allow_redirects=True)
            open(img_path, 'wb').write(r.content)

class ScryfallDraftCreator(BaseDraftCreator):
    def read_pack(self,pack,verbosity=2):
        '''

        '''
        if len(pack.shape) != 1:
            raise Exception('pack must be 1 dimensional')
        
        card_infos = []
        for pack_idx in pack:
            card_infos.append(self.get_info_from_idx(pack_idx))

        card_infos = sorted(card_infos,key=lambda x: self.rarity_ordering[x['Rarity']])
        for card_info in card_infos:
            print(self.str_from_info(card_info,verbosity=verbosity))
    

    def get_info_from_idx(self,idx):
        card_info = self.set_df.iloc[idx]
        if pd.isna(card_info['Mana Cost']):
            mana_cost = '~'
        else:
            mana_cost = card_info['Mana Cost']
        return {'Name':card_info['Name'],
                'Mana Cost':mana_cost,
                'Mana Value':card_info['Mana Value'],
                'Rarity':card_info['Rarity'],
                'Type':card_info['Type']}

    def str_from_info(self,card_info,verbosity=2):
        card_str = ''
        if verbosity >= 0:
            card_str = card_info['Name'] +card_str
        if verbosity >= 1:
            card_str = '{:25}'.format(card_info['Mana Cost']) +card_str
        if verbosity >= 2:
            card_str = '{:3}'.format(card_info['Rarity'][0].upper()) + card_str
        return card_str

    def read_pool_picks(self,bot_pick_vec,verbosity=2):
        raise NotImplementedError


class STXDraftCreator(ScryfallDraftCreator):
    def __init__(self,set_df=None):
        self.set_code = 'stx'
        self.pack_size=15
        #numbers taken from 17L data
        self.common_lesson_rate = 0.924774665362472
        self.rare_lesson_rate = 0.06804351108456
        self.mythic_lesson_rate = 0.00718182355296759
        self.uncommon_lesson_swap_rate = 0.1865764233993527

        self.uncommon_sta_rate = 0.6666819843890527
        self.rare_sta_rate = 0.2661344852260567
        self.mythic_sta_rate = 0.0671835303848906

        self.rare_stx_rate = 0.8635650467081121
        self.mythic_stx_rate = 0.13643495329188796

        self.num_common_slot = 9
        self.num_uncommon_slot = 3
        # and 1 lesson slot, 1 rare slot, 1 sta slot

        if set_df is None:
            self.set_df = self._get_default_set_df()
        else:
            self.set_df = set_df

        self.build_set()
        
    def _get_default_set_df(self):
        return pd.read_csv(pjoin(metadata_path,'STX_card_names.csv'))
    
    def _slot_type_from_row(self,row):
        name = row['Name']
        rarity = row['Rarity'][0] #first initial of rarity: c,u,r,m
        type = row['Type']
        is_lesson = type.endswith('Lesson')
        set_code = row['Set Code']
        
        if  name.lower() in basic_land_names:
            return 'basic_land_'+rarity
        elif is_lesson:
            return 'lesson_'+rarity
        else:
            if set_code == 'sta':
                return 'sta_'+rarity
            elif set_code == 'stx':
                return 'stx_'+rarity
            else:
                raise ValueError

    def build_set(self):
        prefixes = ['stx','sta','lesson','basic_land']
        rarities = ['c','u','r','m']

        self.idx_dict = {}
        for prefix,rarity in product(prefixes,rarities):
            self.idx_dict[prefix+'_'+rarity] = []
        for row_idx,row in self.set_df.iterrows():
            self.idx_dict[self._slot_type_from_row(row)].append(row_idx)

    def create_pack(self):
        """[summary]
        
        See https://www.lethe.xyz/mtg/collation/stx.html for pack generation 
        (Disregard foil changes in probability, MTGA does not use this mechanic)

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """        
        pack = []
        #add commons
        pack.extend(np.random.choice(self.idx_dict['stx_c'], size=self.num_common_slot, replace=False))
        
        #add the lesson slot
        lesson_draw = np.random.multinomial(1,[self.common_lesson_rate,self.rare_lesson_rate,
                                                    self.mythic_lesson_rate])
        if np.argmax(lesson_draw) == 0:
            pack.extend(np.random.choice(self.idx_dict['lesson_c'], size=1, replace=False))
        elif np.argmax(lesson_draw) == 1:
            pack.extend(np.random.choice(self.idx_dict['lesson_r'], size=1, replace=False))
        elif np.argmax(lesson_draw) == 2:
            pack.extend(np.random.choice(self.idx_dict['lesson_m'], size=1, replace=False))

        #add 2 uncommons
        pack.extend(np.random.choice(self.idx_dict['stx_u'], size=self.num_uncommon_slot-1, replace=False))
        #for the 3rd uncommon, theres a chance it gets replaced with a lesson uncommon
        if np.random.uniform() < self.uncommon_lesson_swap_rate:
            pack.extend(np.random.choice(self.idx_dict['lesson_u'],size=1,replace=False))
        else:
            pack.extend(np.random.choice(self.idx_dict['stx_u'],size=1,replace=False))

        #add rare slot
        if np.random.uniform() < self.mythic_stx_rate:
            pack.extend(np.random.choice(self.idx_dict['stx_m'],size=1,replace=False))
        else:
            pack.extend(np.random.choice(self.idx_dict['stx_r'],size=1,replace=False))
        
        #add sta slot
        sta_draw = np.random.multinomial(1,[self.uncommon_sta_rate,self.rare_sta_rate,
                                                    self.mythic_sta_rate])
        if np.argmax(sta_draw) == 0:
            pack.extend(np.random.choice(self.idx_dict['sta_u'], size=1, replace=False))
        elif np.argmax(sta_draw) == 1:
            pack.extend(np.random.choice(self.idx_dict['sta_r'], size=1, replace=False))
        elif np.argmax(sta_draw) == 2:
            pack.extend(np.random.choice(self.idx_dict['sta_m'], size=1, replace=False))

        return np.asarray(pack)

    def read_pool_picks(self,bot_pick_vec,verbosity=2):
        '''
            bot_pick_vec should be of shape (set_size),
            with the value of each idx
        '''

        non_zeros = np.nonzero(bot_pick_vec)[0]
        main_deck = []
        lessons = []
        for card_idx in non_zeros:
            num_card = str(int(bot_pick_vec[card_idx]))
            card_info = self.get_info_from_idx(card_idx)
            card_str = self.str_from_info(card_info,verbosity=verbosity)
            card_str = '{:3}'.format(num_card) + card_str
            if card_info['Type'].endswith('Lesson'):
                lessons.append((card_info['Mana Value'],card_str))
            else:
                main_deck.append((card_info['Mana Value'],card_str))

        main_deck = sorted(main_deck,key=lambda x:x[0])
        lessons = sorted(lessons,key=lambda x:x[0])

        out_str = ''
        for _,card_str in main_deck:
            out_str += card_str + '\n'
        out_str += 'Lessons: \n'
        for _,card_str in lessons:
            out_str += card_str + '\n'

        return out_str


class M19DraftCreator(BaseDraftCreator):

    def __init__(self,set_df=None,land_df=None):
        self.set_code = 'm19'
        self.mythic_prob = 1/8
        self.num_common = 10
        self.num_uncommon = 3
        self.num_rare = 1
        self.num_land = 1
        self.pack_size=15
        
        if set_df is None:
            self.set_df = self._get_default_set_df()
        else:
            self.set_df = set_df
        if land_df is None:
            self.land_df = self._get_default_land_df()
        else:
            self.land_df = land_df

        self.build_set()

    def _get_default_set_df(self):
        return pd.read_csv(pjoin(metadata_path,'M19_standardized_rating.tsv'),delimiter="\t")
    def _get_default_land_df(self):
        return pd.read_csv(pjoin(metadata_path,'M19_land_rating.tsv'),delimiter="\t")

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

        basic_land_names = set(['Plains','Island','Swamp','Mountain','Forest'])
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

    def get_info_from_idx(self,card_idx):
        return self.name_dict[self.label_encoder.inverse_transform([card_idx])[0]]

    def card_name_from_idx(self,card_idx):
        return self.label_encoder.inverse_transform([card_idx])[0]

    def card_and_cc_from_idx(self,card_idx):
        card_info = self.name_dict[self.label_encoder.inverse_transform([card_idx])[0]]
        return

        
    def read_pool_picks(self,bot_pick_vec,verbosity=2):
        '''
            bot_pick_vec should be of shape (set_size),
            with the value of each idx
        '''

        non_zeros = np.nonzero(bot_pick_vec)[0]
        deck = []
        for card_idx in non_zeros:
            num_card = str(int(bot_pick_vec[card_idx]))
            card_info = self.get_info_from_idx(card_idx)
            cast_cost = card_info['Casting Cost 1']
            cmc = self.cmc_from_string(cast_cost)

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

        out_str = ''
        for cmc,card_str in deck:
            out_str += card_str + '\n'
            # print(card_str)
        return out_str

    @staticmethod
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


class BaseSetMetaData:
    draft_creator = None
    @staticmethod
    def load_draft_creator(self):
        raise NotImplementedError
    
class M19MetaData(BaseSetMetaData):
    draft_creator = M19DraftCreator
    set_df_path = pjoin(metadata_path,'M19_standardized_rating.tsv')
    land_df_path = pjoin(metadata_path,'M19_land_rating.tsv')
    image_folder_path = pjoin(images_path,'m19')
    @classmethod
    def load_draft_creator(cls):
        set_df = pd.read_csv(cls.set_df_path,delimiter="\t")
        land_df = pd.read_csv(cls.land_df_path,delimiter="\t")
        return M19DraftCreator(set_df=set_df,land_df=land_df)

class STXMetaData(BaseSetMetaData):
    draft_creator = STXDraftCreator
    set_df_path = pjoin(metadata_path,'STX_card_names.csv')
    image_folder_path = pjoin(images_path,'stx')
    @classmethod
    def load_draft_creator(cls):
        set_df = pd.read_csv(cls.set_df_path)
        return STXDraftCreator(set_df=set_df)

def get_set_metadata(set_code):
    return set_metadata_map[set_code.lower()]

available_sets = ['m19','stx']
set_metadata_map = {'m19':M19MetaData,
                'stx':STXMetaData}