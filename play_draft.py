import tkinter as tk
from PIL import Image, ImageTk
from set_utils import available_sets,set_metadata_map,images_path
import os
from os.path import join as pjoin
from tkinter.filedialog import askopenfilename
import keras
from nn_utils import NNBot
import numpy as np
from functools import partial
import argparse

def resize_image(image, maxsize):
    r1 = image.size[0]/maxsize[0] # width ratio
    r2 = image.size[1]/maxsize[1] # height ratio
    ratio = max(r1, r2)
    newsize = (int(image.size[0]/ratio), int(image.size[1]/ratio))
    image = image.resize(newsize, Image.ANTIALIAS)
    return image

class MainApplication(tk.Frame):
    def __init__(self, root,flag_set=None,flag_model=None):
        tk.Frame.__init__(self, root,width=500, height=500,)
        self.root = root

        #constants
        self.blank_image_path = pjoin(images_path,'blank.jpg')
        self.black_image_path = pjoin(images_path,'black.jpg')
    
        self.display_row = 0
        self.num_bots = 7
        self.num_packs = 3

        #Variables
        self.mtg_set = tk.StringVar(self)
        self.mtg_set.set(available_sets[0])
        self.card_image_width = tk.IntVar(self)
        self.card_image_width.set(244)
        self.card_image_height = tk.IntVar(self)
        self.card_image_height.set(320)

        self.deck_frame_width = tk.IntVar(self)
        self.deck_frame_width.set(320)

        self.menubar = tk.Menu(self)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(
            label="New Draft", command=self.popNewDraftWindow
        )
        filemenu.add_command(
            label="Exit", command=self.popQuitWindow
        )

        options_menu = tk.Menu(self.menubar, tearoff=0)
        # options_menu.add_command(
        #     label="New Draft", command=self.popNewDraftWindow
        # )
        # options_menu.add_command(
        #     label="Exit", command=self.popQuitWindow
        # )

        self.menubar.add_cascade(label="File", menu=filemenu)
        self.menubar.add_cascade(label="Options", menu=options_menu)
        self.root.config(menu=self.menubar)

        
        self.init_main_window()
        self.pack_propagate(0)
        self.focus_set()

        #debug
        if flag_set is not None and flag_model is not None:
            self.model_path=flag_model
            self.mtg_set.set(flag_set)
            self.init_set()
        return
    
    def init_main_window(self):
        self.display_label = tk.Label(self, text="Click 'File > New Draft' to start")
        # self.display_label.grid(row=self.display_row,column=0)
        self.display_label.pack(fill=tk.BOTH, expand=1)
        # self.current_set_label = tk.Label(self, text="Current Set : {}".format(self.mtg_set.get()))
        # self.current_set_label.grid(row=1,column=0)
        self.apply_options()
    
    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    
    def apply_options(self):
        blank_image = Image.open(self.blank_image_path)
        blank_image = resize_image(blank_image,
                                            (self.card_image_width.get(),
                                            self.card_image_height.get())
                                        )
        self.blank_image = ImageTk.PhotoImage(blank_image)
        black_image = Image.open(self.black_image_path)
        black_image = resize_image(black_image,
                                            (self.card_image_width.get(),
                                            self.card_image_height.get())
                                        )
        self.black_image = ImageTk.PhotoImage(black_image)

    def init_set(self):
        

        set_code = self.mtg_set.get()

        if set_code =='m19' or  set_code =='stx':
            self.num_card_rows = tk.IntVar(self)
            self.num_card_rows.set(3)
            self.num_card_cols = tk.IntVar(self)
            self.num_card_cols.set(5)
            self.deck_col = 5
        
        self.set_metadata = set_metadata_map[set_code]
        self.draft_creator = self.set_metadata.load_draft_creator()
        self.set_size = self.draft_creator.get_set_size()
        self.set_images_folder = pjoin(images_path,set_code)


        if hasattr(self,'card_buttons'):
            for card_button in self.card_buttons:
                card_button.destroy()

        self.card_buttons = []
        card_idx = 0
        for row in range(self.num_card_rows.get()):
            for col in range(self.num_card_cols.get()):
                card_button = tk.Button(self, image = self.blank_image,
                                        command=partial(self.card_click,card_idx))
                card_button.grid(row=row+1,column=col,)
                self.card_buttons.append(card_button)
                card_idx+=1
        
        self.deck_frame = tk.Frame(self,width=self.deck_frame_width.get())
        self.deck_frame.grid(rowspan=self.num_card_rows.get(),
                            row=0,
                            column=self.deck_col)
        self.deck_display = tk.Label(self.deck_frame, text="AAAAAAAAAA")
        self.deck_display.grid(row=0,column=0)

        self.display_label.grid(row=self.display_row,column=(self.num_card_cols.get()//2))

        #TODO differentiate between begin_draft and restart_draft
        #restarting draft means the old buttons should be destroyed so as to 
        # not create a mem leak
        #TODO: Above can be done by conceptualizing the set selection, and set restart separately
        #TODO: build deck, deck label
        #TODO: draft cards and images

        self.start_draft()
    
    def update_display(self):
        display_text = '''Drafting : {} \n Pack # : {} \n Pick # : {} \n Pass Direction : {} \n   
                        '''.format(self.mtg_set.get(),
                                self.pack_num,
                                self.pick_num,
                                self.pass_direction
                        )
        self.display_label.configure(text=display_text)
        if np.sum(self.player_deck) ==0:
            self.deck_display.configure(text='No Cards Selected')
        else:
            card_idxs = np.nonzero(self.player_deck)[0]
            deck_str  = ''
            for card_idx in card_idxs:
                deck_str += self.draft_creator.set_df.iloc[card_idx]['Name'] + '\n'
            self.deck_display.configure(text=deck_str)
            return

    
    def start_draft(self):
        self.load_model()

        self.bot_list = []
        for _ in range(self.num_bots):
            nnbot = NNBot(keras_model=self.model,
                        set_size=self.set_size,
                        pick_mode='max'
                        )
            self.bot_list.append(nnbot)

        self.reset_picks()
        
        self.pack_num = 0
        self.pick_num = 0
        self.set_pick_direction()
        self.create_packs()
        self.pick_rotation()
    
    def create_packs(self):
        self.bot_packs = []
        for _ in range(self.num_bots):
            self.bot_packs.append(self.draft_creator.create_pack())
        self.player_pack = self.draft_creator.create_pack()

    def reset_picks(self):
        self.player_deck = np.zeros((self.set_size))
        for bot in self.bot_list:
            bot.reset_picks()

    def set_pick_direction(self):
        if self.pack_num %2 == 0:
            self.pass_dir_val = -1
            self.pass_direction = 'left'
        else:
            self.pass_dir_val = 1
            self.pass_direction = 'right'

    
    def pick_rotation(self):
        self.update_display()

        if self.pick_num == self.draft_creator.pack_size:
            #TODO update pack num
            pass

        #bot picks
        self.new_bot_packs = [[] for _ in range(self.num_bots)]
        for bot_idx,nnbot in enumerate(self.bot_list):
            pack = self.bot_packs[bot_idx]
            out_pack = nnbot.pick_and_add(pack)
            out_bot_idx = bot_idx + self.pass_dir_val
            if out_bot_idx == -1 or out_bot_idx == self.num_bots+1:
                self.new_player_pack = out_pack
            else:
                self.new_bot_packs[out_bot_idx] = out_pack

        #set up player pick
        self.show_player_cards()

    def show_player_cards(self):
        card_infos = []
        for card_idx in self.player_pack:
            if card_idx == -1:
                continue
            card_infos.append((card_idx,self.draft_creator.get_info_from_idx(card_idx)))

        self.player_card_tuples = sorted(card_infos,key=lambda x: -self.draft_creator.rarity_ordering[x[1]['Rarity']])
        self.card_photoimages = []
        for button_idx,_ in enumerate(self.player_card_tuples):
            if button_idx < len(self.player_card_tuples):
                card_tuple = self.player_card_tuples[button_idx]
                card_idx,card_info = card_tuple
                print(self.draft_creator.str_from_info(card_info))

                image_path = pjoin(self.set_images_folder,'{:04d}.jpg'.format(card_idx))
                card_image = Image.open(image_path)
                card_image = resize_image(card_image,
                                                (self.card_image_width.get(),
                                                self.card_image_height.get())
                                            )
                card_photoimage = ImageTk.PhotoImage(card_image)
                self.card_buttons[button_idx].config(image=card_photoimage)
                self.card_photoimages.append(card_photoimage)
            else:
                self.card_buttons[button_idx].config(image=self.black_image)
        #on_command, pick card, rotate cards and run pick_rotation

    def add_player_card(self,card_idx):
        self.player_deck[card_idx]+=1

    def card_click(self,button_idx):
        print("{}".format(button_idx))
        if button_idx >= len(self.player_card_tuples):
            return
        card_idx,card_info = self.player_card_tuples[button_idx]
        self.add_player_card(card_idx)

        self.player_pack[button_idx] = -1

        if self.pass_dir_val == -1:
            self.new_bot_packs[self.num_bots-1] = self.player_pack
        elif self.pass_dir_val == 1:
            self.new_bot_packs[0] = self.player_pack
        self.player_pack = self.new_player_pack
        self.bot_packs = self.new_bot_packs
        self.pick_rotation()
    
    def popQuitWindow(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = QuitWindow(self.newWindow, self)

    def popNewDraftWindow(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = newDraftWindow(self.newWindow, self)

    
    def reset(self):
        self.root.update_idletasks()
        return

class QuitWindow(tk.Frame):
    def __init__(self, parent, main, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.mainApp = main
        self.label = tk.Label(self, text="Close Program?")
        self.label.pack()
        self.quitButton = tk.Button(
            self, text="Yes", width=25, command=self.quit
        )
        self.quitButton.pack(side='left')
        self.returnButton = tk.Button(
            self, text="No", width=25, command=self.return_to
        )
        self.returnButton.pack(side='right')
        self.pack()

    def return_to(self):
        self.parent.destroy()
    def quit(self):
        self.mainApp.parent.destroy()

class newDraftWindow(tk.Frame):
    def __init__(self, parent, main, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.mainApp = main

        label = tk.Label(self, text="Choose Set")
        label.pack()
        w = tk.OptionMenu(self, main.mtg_set, *available_sets)
        w.pack()

        button = tk.Button(self, text="OK", command=self.choose_model)
        button.pack()

        self.pack()

    def choose_model(self):
        model_path = askopenfilename(initialdir = "/",
                                    title = "keras model hdf5 file",
                                    filetypes = (("hdf5","*.hdf5"),("all files","*.*")))
        self.mainApp.model_path =model_path
        self.return_to()

    def return_to(self):
        # print(self.mainApp.mtg_set.get())
        self.mainApp.reset()
        self.mainApp.init_set()
        self.parent.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_hdf5", action="store", dest="model_hdf5",
        default=None,
        help=("path to the model_hdf5 ")
    )
    parser.add_argument(
        "--set_code", action="store", dest="set_code",
        default=None,
        help=("3-symbol code for the set, (only 'm19' and 'stx' available atm) ")
    )
    args = parser.parse_args()
    root = tk.Tk()
    MainApplication(root,flag_set=args.set_code,flag_model=args.model_hdf5).pack(fill="both", expand=True)
    root.mainloop()
