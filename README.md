# mtg_draft_bot

Code for training and running a simple neural network bot which will draft Magic The Gathering sets. Allows user to simulate a draft of 8 bots, or use a GUI to draft against 7 bots.

The model I use in this repo is similar to the neural network model in the paper [https://arxiv.org/abs/2009.00655](https://arxiv.org/abs/2009.00655). I use tensorflow-keras model instead of pytorch.

Currently supported sets: M19 and STX, will add more more sets if data for those sets becomes available.

## Requirements
The versions are not hard requirements. They are just what I use.
* tensorflow>=2.3.0
* keras==2.4.3
* pandas==1.1.2
* scikit-learn==0.23.2
* Pillow==5.4.1

## Setting Up the Data & Training the Bots
Since the M19 data (from draftsim) and the STX data (from 17Lands) are formatted differently, the procedure for each of them will be slightly different.

### Pre-Trained Models
Here are some pre-trained models, if you want to skip the training steps below.

[M19 Model](https://drive.google.com/file/d/1YplrfTjG31SiKLTtg2g19aB5cZYb-qKA/view?usp=sharing)

[STX Premier Draft Model](https://drive.google.com/file/d/1FurL_mpAPQR_8Mj7NDRo6rfqF7biHBHy/view?usp=sharing)

[STX Traditional Draft Model](https://drive.google.com/file/d/1BQUt-MEi-SBYziPUu0MvhfE-HEtxAuIF/view?usp=sharing)

### Draftsim Data (M19)
Download data found here [https://draftsim.com/draft-data/](https://draftsim.com/draft-data/)

1. Unzip the data downloaded above, this will give you a data_folder containing all the draft data
2. run 'python create_draftsim_draft_data.py --data_folder {path_to_data_folder}'. This might take awhile as it preprocesses the data.
3 files will be created, a train_drafts.pkl,test_drafts.pkl,and standardized_m19_rating.tsv. You will use these pkls in the next step. You should check to see if the generated standardized_m19_rating.tsv file matches the one in the mtg_draft_bot/metadata folder.
3. run  'python train_nn.py --train_pkl {} --test_pkl {} --output_name {}' . The first 
2 files are the outputs from step 2, 'output_name' is the path to the output hdf5 file that you want to create, this contains the model

### 17Lands Data (STX)
The data can be found here [https://www.17lands.com/public_datasets](https://www.17lands.com/public_datasets). You can use either the 'STX Premier Draft Data' or the 'STX Traditional Draft Data'; this data is collected from the draft process. (Do not use 'STX Traditional/Premier Draft Game Data', which is data from the games played)
1. Unzip the data downloaded above, in here will be a 'data_csv' which will be used in the next step
2. run 'python create_17lands_draft_data.py --data_csv {path_to_data_csv}'. This might take awhile as it preprocesses the data.
3 files will be created, a _train_data.pkl,_test_data.pkl,and a _card_names.csv. You will use these pkls in the next step. You should check to see if the card named in each row in the generated _card_names.csv file matches 'STX_card_names.csv' in the mtg_draft_bot/metadata folder. (Note: the information stored in the two csvs will be different, just ensure that each row corresponds to the same card)
3. run  'python train_nn.py --train_pkl {} --test_pkl {} --output_name {}' . The first 
2 files are the outputs from step 2, 'output_name' is the path to the output hdf5 file that you want to create, this contains the model

## Playing a Draft Against Bots
1. Run 'python play_draft.py'
2. On the top bar, click:  File > New Draft
3. Choose the set you want to draft, then press OK.
4. Use the file selector to select the model hdf5 file (must correspond to the set chosen in step 3)
5. Now you should see the cards, left click on one to add it to your deck. The bots will pick their cards and then you move onto your next pick

Here is an example of what it should look like:
![Example Image](https://github.com/PhilippeNguyen/mtg_draft_bot/blob/master/assets/play_draft_ex_1.png)

## Simulating an All Bot Draft
4. To run a simulation draft, run 'python simulate_draft.py --model_hdf5 {} --set_code {}. 
The model_hdf5 is the hdf5 model that was trained in the training stage, the set_code is 'm19' or 'stx'


Once you run it you should see what decks each bot drafted like:

```
M19 Example:

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

STX Example:

Bot_Num: 7
3  C  ~                        Archway Commons
1  C  ~                        Prismari Campus
2  C  ~                        Quandrix Campus
1  C  ~                        Witherbloom Campus
1  C  {1}{G}                   Big Play
1  C  {1}{U}                   Curate
2  U  {G}{U}                   Decisive Denial
1  U  {1}{B}                   Eliminate
2  U  {U}{R}                   Expressive Iteration
1  R  {G}{U}                   Growth Spiral
1  C  {G}{U}                   Needlethorn Drake
1  U  {1}{U}                   Negate
1  U  {G}{U}                   Quandrix Apprentice
2  C  {1}{G/U}                 Square Up
1  U  {1}{U}                   Strategic Planning
2  C  {1}{G}                   Tangletrap
3  C  {U/R}{U/R}               Teach by Example
1  R  {3}                      Codie, Vociferous Codex
1  U  {2}{R}                   Igneous Inspiration
1  C  {3}                      Letter of Acceptance
1  C  {2}{G}                   Mage Duel
1  R  {3}{G}                   Accomplished Alchemist
1  C  {4}                      Biblioplex Assistant
1  C  {2}{G}{U}                Eureka Moment
1  C  {2}{B}{G}                Moldering Karok
1  U  {2}{B}{G}                Mortality Spear
1  C  {3}{U}                   Serpentine Curve
1  C  {3}{U}                   Waterfall Aerialist
2  C  {4}{U}                   Bury in Books
1  U  {3}{U}{R}                Practical Research
1  C  {5}{G}                   Leyline Invocation
2  U  {3}{U}{U}{R}{R}          Creative Outburst
Lessons:
1  C  {2}                      Environmental Sciences
1  C  {1}{B/G}{B/G}            Pest Summoning
```
