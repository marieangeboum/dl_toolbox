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
import matplotlib.pyplot as plt
import dl_toolbox.inference as dl_inf
from sklearn.utils import shuffle
from collections import defaultdict, Counter
from argparse import ArgumentParser
from dl_toolbox.torch_datasets import *
from dl_toolbox.torch_datasets.utils import *
from datetime import datetime, time

parser = ArgumentParser()
parser.add_argument("--data_path", type=str, default ='/scratchf/CHALLENGE_IGN/FLAIR_1/train')
parser.add_argument("--metadata", type=str, default = 'flair-one_metadata.json')
parser.add_argument("--target_camera", type=str, default ='UCE')
parser.add_argument("--target_year", type=str, default ='2020')
parser.add_argument("--target_zone", type=str, default ='UU')
parser.add_argument("--time_slot", nargs=2, type=float, default=(8,11))
args = parser.parse_args()

target_zone = args.target_zone
target_camera = args.target_camera
target_year = args.target_year
time_slot = args.time_slot
file_path = args.metadata

# Define the format of the time string
time_format = '%Hh%M'

start_time_slot = time(args.time_slot[0],0)
end_time_slot = time(args.time_slot[1],0)
def is_time_in_timeslot(check_time, start_time, end_time):
    check_datetime = datetime.combine(datetime.today(), check_time)
    start_datetime = datetime.combine(datetime.today(), start_time)
    end_datetime = datetime.combine(datetime.today(), end_time)
    return start_datetime <= check_datetime <= end_datetime

try:
    with open(file_path, "r") as file:
        metadata = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error decoding JSON data from '{file_path}'.")

# Use dictionary comprehension to create a new dictionary with the original keys
filtered_data = {key: value for key, value in metadata.items() if value['zone'].split('_')[1] == target_zone 
                 and value['camera'].split('-')[0] == target_camera and value['date'].split('-')[0] == target_year and  is_time_in_timeslot(datetime.strptime(value['time'], time_format).time(), start_time_slot, end_time_slot) == True}
# Get images from filtered data dict
imgs_name_dataset = list(filtered_data.keys())

# Create a new dictionary to store sets of unique values for each key
unique_values_filtered_data = {}
# Iterate through the dictionaries and populate the unique_values_dict
for sub_dict in filtered_data.values():
    for key, value in sub_dict.items():
        if key == 'zone':
            value = value.split('_')[1]
        if key not in unique_values_filtered_data:
            unique_values_filtered_data[key] = set()
        unique_values_filtered_data[key].add(value)

#Get domains list
domains_list = list(unique_values_filtered_data['domain'])
# Create a dictionary to store the count of occurrences for each unique value
occurrence_counts = defaultdict(lambda: defaultdict(int))

# Iterate through the filtered_data and count occurrences for each key-value pair
for sub_dict in filtered_data.values():
    for key, value in sub_dict.items():
        # print(key)
        if key == 'zone':
            value = value.split('_')[1]          
        occurrence_counts[key][value] += 1    

# Extract keys and values from the dictionary
labels = list(occurrence_counts['domain'].keys())
values = list(occurrence_counts['domain'].values())
# Create a bar plot
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.bar(labels, values)
# Add labels and title
plt.xlabel('Domains')
plt.ylabel('Nb of Images')
plt.title('"{}" Zones Image Distribution in {} with {} sensors \n between {} and {}'.format(target_zone, target_year, target_camera, start_time_slot, end_time_slot))
# Rotate x-axis labels for better visibility
plt.xticks(rotation=45)
# Show the plot
plt.tight_layout()  # Ensures all labels are visible
plt.show()        

.



