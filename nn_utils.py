import keras
import keras.layers as KL
import numpy as np
import scipy

def build_model(input_size, num_dense=3):
    input_layer = KL.Input(shape=(input_size,))
    set_size = input_size // 2
    deck = KL.Lambda(lambda x: x[:,:set_size],output_shape=(set_size,))(input_layer)
    cur_pick = KL.Lambda(lambda x: x[:,set_size:],output_shape=(set_size,))(input_layer)

    x = deck
    for idx in range(num_dense):
        x = KL.Dense(set_size)(x)
        x = KL.ReLU()(x)
        x = KL.BatchNormalization()(x)

    x = KL.Dense(set_size)(x)
    masked_logits = KL.Lambda(lambda x: (x[0]* x[1]))([x,cur_pick])

    model = keras.models.Model(input_layer,masked_logits)
    return model


class NNBot(object):
    def __init__(self,keras_model,set_size,pick_mode='max',invalid_mode=-1):
        '''
            Note the NNBot handles the already picked card values differently from
            the training code/pkls. picked cards will be set to -1, instead of 0. This
            means that the NNBot doesn't need to keep track of the pick num
            TODO: handle the training data of picked cards
        '''
        self.keras_model = keras_model
        self.set_size = set_size
        self.pick_mode = pick_mode
        self.invalid_mode = invalid_mode
        self.reset_picks()

    def reset_picks(self):
        self.picks = np.zeros((self.set_size))

    def pack_probs(self,pack):
        model_input = self.pack_to_model_input(pack)
        out = self.keras_model.predict(np.expand_dims(model_input,0))
        return np.squeeze(out,axis=0)
    
    def pick(self,pack):
        out = self.pack_probs(pack)
        if self.pick_mode == 'max':
            return np.argmax(out)
        elif self.pick_mode == 'sample':
            probs = scipy.special.softmax(out)
            probs[probs<5e-4] = 0
            pick = np.argmax(np.random.multinomial(1,probs))
            while pick not in pack:
                print('pick not found in pack,trying again')
                pick = np.argmax(np.random.multinomial(1, probs))
            return pick

    def pick_and_add(self,pack):
        card_idx = self.pick(pack)
        self.add(card_idx)
        out_pack = pack.tolist()
        out_pack.remove(card_idx)
        out_pack.append(-1)
        return np.asarray(out_pack)

    def add(self,card_idx):
        self.picks[card_idx]+=1

    def pack_to_vec(self,pack):
        vec = np.zeros((self.set_size))
        for card_idx in pack:
            if self.invalid_mode == -1:
                if card_idx != -1:
                    vec[card_idx] = 1
            else:
                raise NotImplementedError('unknown invalid_mode')
        return vec
    
    def pack_to_model_input(self,pack):
        vec = self.pack_to_vec(pack)
        model_input = np.concatenate((self.picks,vec))
        return model_input

