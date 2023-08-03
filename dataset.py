import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import glob
import numpy as np
import time
import tabulate
import fnmatch
import random
import rasterio
import subprocess

import dl_toolbox.inference as dl_inf

from sklearn.utils import shuffle
from collections import defaultdict, Counter
from argparse import ArgumentParser
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default ='/scratchf/CHALLENGE_IGN/FLAIR_1/train')
parser.add_argument("--json_metadata", type=str, default = '/scratchf/CHALLENGE_IGN/FLAIR_1/flair-one_metadata.json')
parser.add_argument("--target_camera", type=str, default ='UCE')
parser.add_argument("--target_year", type=str, default ='2021')
parser.add_argument("--target_zone", type=str, default ='UN')
args = parser.parse_args()

target_zone = 'UN'
target_camera = 'UCE'
target_year = '2021'
# Step 1: Specify the file path
file_path = args.json_metadata
try:
    # Step 2: Open the JSON file in read mode
    with open(file_path, "r") as file:
        # Step 3: Load JSON data into a Python data structure
        data = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error decoding JSON data from '{file_path}'.")

# Use dictionary comprehension to create a new dictionary with the original keys
filtered_data = {key: value for key, value in data.items() if value['zone'].split('_')[1] == target_zone and value['camera'].split('-')[0] == target_camera and value['date'].split('-')[0] == target_year}

    
    