import json
import random
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client
import requests
import seaborn
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kfold import Kfold
from mlp_multiclass import FFNeuralNetwork, He_, Layer, Neuron, Xavier
from temporal_classification import (classify_temporal_series,
                                     plot_real_trajectory,
                                     plot_temporal_classification)
