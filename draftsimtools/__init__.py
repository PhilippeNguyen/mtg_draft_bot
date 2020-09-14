"""
  draftsimtools
  ~~~~~~~~~~~~~

  Utilies for building machine learning bots from draftsim data.
"""

import pandas as pd

from .load import *
from .bot import *
from .bot_tester import *
from .random_bot import *
from .raredraft_bot import *
from .classic_bot import *
from .nnet_architecture import *
from .nnet_bot import *
from .bayes_bot import *

# from . import bots # As of Aug 5, generates an error. Is it obsolete now?

#from .bots/draftsim_bot_2018 import *
#from .bots/draftsim_bot_2018_sgd import *
