import json
from collections import defaultdict, Counter
# Step 1: Specify the file path
file_path = "/d/maboum/dl_toolbox/flair-one_metadata.json"

try:
    # Step 2: Open the JSON file in read mode
    with open(file_path, "r") as file:
        # Step 3: Load JSON data into a Python data structure
        data = json.load(file)
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except json.JSONDecodeError:
    print(f"Error decoding JSON data from '{file_path}'.")
# {'D068_2021', 'D033_2021', 'D067_2021', 'D004_2021', 'D030_2021', 'D080_2021', 'D029_2021', 'D022_2021', 'D064_2021', 'D060_2021', 'D075_2021', 'D038_2021', 'D066_2021'}
# target_domain = 'D067_2021'
target_zone = 'UN'
target_camera = 'UCE'
target_year = '2021'

# Use dictionary comprehension to create a new dictionary with the original keys
filtered_data = {key: value for key, value in data.items() if value['zone'].split('_')[1] == target_zone and value['camera'].split('-')[0] == target_camera and value['date'].split('-')[0] == target_year}

# # Create a new dictionary to store sets of unique values for each key
dataset = {}

# Iterate through the dictionaries and populate the unique_values_dict
for sub_dict in filtered_data.values():
    for key, value in sub_dict.items():
        if key == 'zone':
            value = value.split('_')[1]
        if key not in dataset:
            dataset[key] = set()
        dataset[key].add(value)

# Create a dictionary to store the count of occurrences for each unique value
occurrence_counts = defaultdict(lambda: defaultdict(int))

# Iterate through the filtered_data and count occurrences for each key-value pair
for sub_dict in filtered_data.values():
    for key, value in sub_dict.items():
        # print(key)
        if key == 'zone':
            value = value.split('_')[1]          
        occurrence_counts[key][value] += 1

