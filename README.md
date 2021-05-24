# mtg_draft_bot

Code for training a simple neural network bot which will draft the M19 Magic The Gathering Core Set. 
Uses the data from [https://github.com/khakhalin/MTG](https://github.com/khakhalin/MTG) (some of their processing code is used here in the draftsimtools subfolder).
The paper can be found here [https://arxiv.org/abs/2009.00655](https://arxiv.org/abs/2009.00655), and the model I use in this
repo is similar to the neural network model in the paper. I use tensorflow-keras model instead of pytorch.

Currently, code here is just for training the NN model and running a simulation draft with 8
copies of that model. You can see an example of the simulation below.

Currently supported sets: M19 and STX, will add more if data becomes available.

## Requirements
The versions are not hard requirements. They are just what I use.
* tensorflow>=2.3.0
* keras==2.4.3
* pandas==1.1.2
* scikit-learn==0.23.2

Download data found here [https://draftsim.com/draft-data/](https://draftsim.com/draft-data/)

## Usage
1. Unzip the data downloaded above, this will give you a data_folder containing all the draft data
2. run 'python create_draft_data.py --data_folder {path_to_data_folder}'. This might take awhile as it preprocesses the data.
3 files will be created, a train_drafts.pkl,test_drafts.pkl,and standardized_m19_rating.tsv. You will use these in the next step
3. run  'python train_nn.py --train_pkl {} --test_pkl {} --standard_rating_tsv {} --output_name {}' . The first 
3 files are the outputs from step 2, 'output_name' is the path to the output hdf5 file that you want to create, this contains the model
4. To run a simulation draft, run 'python simulate_draft.py --model_hdf5 {} --standard_rating_tsv {} --land_rating_tsv {}'. 
The model_hdf5 is the output from 3, standard_rating_tsv is from 2, and land_rating_tsv is already in the data_folder.

Once you run it you should see what decks each bot drafted like:

```
Bot Num 4

1  C  0         Island
1  C  0         Swamp
1  U  B         Diregraf_Ghoul
1  U  B         Nightmare's_Thirst
1  C  R         Smelt
1  C  1B        Abnormal_Endurance
1  C  1W        Cavalry_Drillmaster
1  C  1B        Child_of_Night
1  C  1W        Daybreak_Chaplain
2  C  1B        Doomed_Dissenter
2  C  1B        Infernal_Scarring
1  C  1W        Mighty_Leap
1  C  1G        Naturalize
1  C  1B        Sovereign's_Bite
1  C  1W        Take_Vengeance
1  C  1B        Walking_Corpse
3  C  2B        Hired_Blade
1  C  2W        Invoke_the_Divine
1  C  2W        Loxodon_Line_Breaker
2  U  2W        Make_a_Stand
1  U  2W        Militia_Bugler
1  C  2B        Mind_Rot
1  U  1BB       Murder
1  C  2B        Skymarch_Bloodletter
1  C  3W        Dwarven_Priest
3  C  3B        Infectious_Horror
1  C  2WW       Inspired_Charge
1  C  3B        Skeleton_Archer
1  C  3W        Star-Crowned_Stag
1  C  3B        Strangling_Spores
1  C  4B        Epicure_of_Blood
1  C  3RR       Fire_Elemental
1  U  3WW       Herald_of_Faith
1  U  4W        Knightly_Valor
2  U  3BB       Vampire_Sovereign
2  C  4BB       Bogstomper

```

Note lands might appear more frequently than in real packs, since I didn't filter out "common" lands
from the "common" slots. I couldn't find much info on card slots in M19 packs, and I'm fairly new to drafting, so 
the simulated pack creation can be fixed once I learn more.