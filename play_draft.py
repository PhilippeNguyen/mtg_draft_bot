import tkinter as tk
from PIL import Image, ImageTk,ImageOps
from set_utils import available_sets,set_metadata_map,images_path
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
        self.canvas_row = 1
        self.canvas_col = 0
        self.scrollbar_col = 1
        self.deck_col = 2
        self.num_bots = 7
        self.num_packs = 3
        self.aspect_ratio = 320/244
        self.init_height = 320
        self.init_width = self.init_height / self.aspect_ratio
        
        
        #Variables
        self.mtg_set = tk.StringVar(self)
        self.mtg_set.set(available_sets[0])
        self.card_image_width = tk.IntVar(self)
        self.card_image_width.set(self.init_width)
        self.card_image_height = tk.IntVar(self)
        self.card_image_height.set(self.init_height)
        self.scale = tk.DoubleVar(self)
        self.scale.set(1.)

        self.deck_frame_width = tk.IntVar(self)
        self.deck_frame_width.set(self.init_width)


        #possible views
        self.view_names = ['Player']
        for idx in range(self.num_bots):
            self.view_names.append('Bot {}'.format(idx+1))
        self.view_map = {name:idx for idx,name in enumerate(self.view_names)}
        self.inv_view_map = {idx:name for idx,name in enumerate(self.view_names)}
        self.cur_view_name = tk.StringVar(self)
        self.cur_view_name.set(self.view_names[0]) #initial view on player
        self.cur_view_idx = tk.IntVar(self)
        self.cur_view_idx.set(0) #initial view on player

        self.menubar = tk.Menu(self)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(
            label="New Draft", command=self.popNewDraftWindow
        )
        filemenu.add_command(
            label="Exit", command=self.popQuitWindow
        )

        options_menu = tk.Menu(self.menubar, tearoff=0)
        options_menu.add_command(
            label="Change Settings", command=self.popChangeSettingsMenu
        )

        view_menu = tk.Menu(self.menubar, tearoff=0)
        view_menu.add_command(
            label="View Bot/Player", command=self.popChangeViewMenu
        )

        self.menubar.add_cascade(label="File", menu=filemenu)
        self.menubar.add_cascade(label="Options", menu=options_menu)
        self.menubar.add_cascade(label="View", menu=view_menu)
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
        self.display_label.pack(fill=tk.BOTH, expand=1)

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
            
        
        self.set_metadata = set_metadata_map[set_code]
        self.draft_creator = self.set_metadata.load_draft_creator()
        self.draft_creator.download_images(verbose=1)
        self.set_size = self.draft_creator.get_set_size()
        self.set_images_folder = pjoin(images_path,set_code)

        if hasattr(self,'deck_frame'):
            self.deck_frame.destroy()
        self.deck_frame = tk.Frame(self,width=self.deck_frame_width.get())
        self.deck_frame.grid(rowspan=self.num_card_rows.get(),
                    row=0,
                    column=self.deck_col)

        if hasattr(self,'deck_display'):
            self.deck_display.destroy()
        self.deck_display = tk.Label(self.deck_frame, text="~",justify='left')
        self.deck_display.grid()


        if hasattr(self,'canvas_scrollbar'):
            self.canvas_scrollbar.destroy()
        self.canvas_scrollbar = tk.Scrollbar(self, orient="vertical")
        self.canvas_scrollbar.grid(row=self.canvas_row,
                                column=self.scrollbar_col,
                                sticky='nse')

        if hasattr(self,'canvas'):
            self.canvas.destroy()
        self.canvas = tk.Canvas(self,yscrollcommand=self.canvas_scrollbar.set,
                                width=self.num_card_cols.get()*self.card_image_width.get(),
                                height=self.num_card_rows.get()*self.card_image_height.get())
        self.canvas.grid(row=self.canvas_row,
                    column=self.canvas_col)

        self.canvas_scrollbar.config(command=self.canvas.yview)
        
        self.canvas_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0), window=self.canvas_frame, 
                                  anchor='nw')

        if hasattr(self,'card_buttons'):
            for card_button in self.card_buttons:
                card_button.destroy()

        self.card_buttons = []
        card_idx = 0
        for row in range(self.num_card_rows.get()):
            for col in range(self.num_card_cols.get()):
                card_button = tk.Button(self.canvas_frame, image = self.blank_image,
                                        command=partial(self.card_click,card_idx),
                                        )
                card_button.grid(row=row,column=col,)
                self.card_buttons.append(card_button)
                card_idx+=1
        
        self.display_label.grid(row=self.display_row,column=self.canvas_col)

        self.start_draft()

    def update_display(self):
        display_text = '''Drafting : {} \n Viewing : {} \n Pack # : {} \n Pick # : {} \n Pass Direction : {} 
                        '''.format(self.mtg_set.get(),
                                self.cur_view_name.get(),
                                self.pack_num+1,
                                self.pick_num+1,
                                self.pass_direction
                        )
        self.display_label.configure(text=display_text)

        cur_view_idx =self.cur_view_idx.get()
        if np.sum(self.all_decks[cur_view_idx]) ==0:
            self.deck_display.configure(text='No Cards Selected')
        else:

            deck_str = self.draft_creator.read_pool_picks(self.all_decks[cur_view_idx])
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

        self.all_decks = [self.player_deck]
        for bot in self.bot_list:
            self.all_decks.append(bot.picks)
        
        self.pack_num = 0
        self.pick_num = 0
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
        
        if self.pick_num == self.draft_creator.pack_size:
            self.pack_num+=1
            self.pick_num=0
            if self.pack_num == self.num_packs:
                self.display_cards()
                return

        if self.pick_num==0:
            self.set_pick_direction()
            self.create_packs()
        
        self.all_packs = [self.player_pack,*self.bot_packs]

        #bot picks
        #TODO: track bot picks
        bot_picks = []
        self.new_bot_packs = [[] for _ in range(self.num_bots)]
        for bot_idx,nnbot in enumerate(self.bot_list):
            pack = self.bot_packs[bot_idx]
            out_pack,card_idx = nnbot.pick_and_add(pack)

            bot_picks.append(card_idx)

            out_bot_idx = bot_idx + self.pass_dir_val
            if out_bot_idx == -1 or out_bot_idx == self.num_bots:
                self.new_player_pack = out_pack
            else:
                self.new_bot_packs[out_bot_idx] = out_pack
        
        self.all_picks = [None,*bot_picks]

        #player pack 
        card_infos = []
        for card_idx in self.player_pack:
            if card_idx == -1:
                continue
            card_infos.append((card_idx,self.draft_creator.get_info_from_idx(card_idx)))

        self.player_card_tuples = sorted(card_infos,key=lambda x: -self.draft_creator.rarity_ordering[x[1]['Rarity']])

        #set up player pick
        self.update_display()
        self.display_cards()

    def display_cards(self):  

        cur_view_idx = self.cur_view_idx.get()
        card_infos = []
        for card_idx in self.all_packs[cur_view_idx]:
            if card_idx == -1:
                continue
            card_infos.append((card_idx,self.draft_creator.get_info_from_idx(card_idx)))

        card_tuples = sorted(card_infos,key=lambda x: -self.draft_creator.rarity_ordering[x[1]['Rarity']])
        card_pick = self.all_picks[cur_view_idx]

        self.card_photoimages = []
        for button_idx,_ in enumerate(card_tuples):
            if button_idx < len(card_tuples):
                card_tuple = card_tuples[button_idx]
                card_idx,card_info = card_tuple

                image_path = pjoin(self.set_images_folder,'{:04d}.jpg'.format(card_idx))
                card_image = Image.open(image_path)
                card_image = resize_image(card_image,
                                                (self.card_image_width.get(),
                                                self.card_image_height.get())
                                            )
                
                if card_pick is not None and card_pick == card_idx:
                    card_image = ImageOps.expand(card_image, border=10, fill=(255,0,255))
                card_photoimage = ImageTk.PhotoImage(card_image)
                self.card_buttons[button_idx].config(image=card_photoimage)
                self.card_photoimages.append(card_photoimage)
            else:
                self.card_buttons[button_idx].config(image=self.black_image)
        
        self.canvas_frame.update_idletasks()
        min_height = self.root.winfo_screenheight()-self.display_label.winfo_height() -80
        self.canvas.configure(
                                width=min(self.root.winfo_screenwidth(),
                                            self.num_card_cols.get()*self.card_image_width.get()),
                                height=min(min_height,
                                        self.num_card_rows.get()*self.card_image_height.get()),
                                )
        
        self.canvas.configure(scrollregion=self.canvas_frame.bbox("all"))
        

    def add_player_card(self,card_idx):
        self.player_deck[card_idx]+=1

    def card_click(self,button_idx):
        if self.cur_view_idx.get() != 0:
            return
        if button_idx >= len(self.player_card_tuples):
            return
        card_idx,card_info = self.player_card_tuples[button_idx]
        self.add_player_card(card_idx)

        pack = self.player_pack.tolist()
        pack.remove(card_idx)
        pack.append(-1)
        self.player_pack =  np.asarray(pack)

        if self.pass_dir_val == -1:
            self.new_bot_packs[self.num_bots-1] = self.player_pack
        elif self.pass_dir_val == 1:
            self.new_bot_packs[0] = self.player_pack
        self.player_pack = self.new_player_pack
        self.bot_packs = self.new_bot_packs

        self.pick_num+=1
        self.pick_rotation()
    
    def popQuitWindow(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = QuitWindow(self.newWindow, self)

    def popNewDraftWindow(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = newDraftWindow(self.newWindow, self)

    def popChangeViewMenu(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = ChangeView(self.newWindow, self)

    def popChangeSettingsMenu(self):
        self.newWindow = tk.Toplevel(self.root)
        self.app = ChangeSettings(self.newWindow, self)

    def reset(self):
        self.root.update_idletasks()
        return


class ChangeSettings(tk.Frame):
    def __init__(self, parent, main, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.mainApp = main
        

        label = tk.Label(self, text="Card Size")
        label.pack()
        self.scale_slider = tk.Scale(self,orient='horizontal',
                                     from_=0.25, to=2.0,
                                     resolution=0.05)
        self.scale_slider.set(self.mainApp.scale.get())
        self.scale_slider.pack()
        button = tk.Button(self, text="OK", command=self.return_to)
        button.pack()
        self.pack()

    def return_to(self):
        self.mainApp.scale.set(self.scale_slider.get())
        self.mainApp.card_image_height.set(self.mainApp.scale.get()*self.mainApp.init_height)
        self.mainApp.card_image_width.set(self.mainApp.scale.get()*self.mainApp.init_width)
        self.mainApp.apply_options()
        self.mainApp.display_cards()
        self.mainApp.update_display()
        self.parent.destroy()


class ChangeView(tk.Frame):
    def __init__(self, parent, main, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.mainApp = main
        label = tk.Label(self, text="Choose View")
        label.pack()
        w = tk.OptionMenu(self, main.cur_view_name, *main.view_names)
        w.pack()
        button = tk.Button(self, text="OK", command=self.return_to)
        button.pack()

        self.pack()

    def return_to(self):
        self.mainApp.cur_view_idx.set(self.mainApp.view_map[self.mainApp.cur_view_name.get()])
        self.mainApp.display_cards()
        self.mainApp.update_display()
        self.parent.destroy()

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
