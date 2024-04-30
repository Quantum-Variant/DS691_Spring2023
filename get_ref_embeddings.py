import os
import numpy as np
import json

# Specify the directory where your .np files are stored
directory = 'data/caltech-101/Batch files'

# List all .npy files in the directory
files = [f for f in os.listdir(directory) if f.endswith('input_pairs.npy')]

# Initialize an empty dictionary to hold the average embeddings of each class
class_averages = {}

# Process each file
for file in files:
    # Extract the class name from the filename
    class_name = file.replace('_input_pairs.npy', '')

    # Load the embeddings from file
    embeddings = np.load(os.path.join(directory, file))
    
    image_embeddings = []
    
    for j in embeddings:
        i = j[0]
        if len(i) == 2:
            i = i[0]
        image_embeddings.append(i)

    # Calculate the average of the image embeddings
    average_image_embeddings = np.mean(image_embeddings, axis=0)

    # Convert numpy array to list for JSON compatibility
    average_list = average_image_embeddings.tolist()

    # Store the average in the dictionary with the class name as the key
    class_averages[class_name] = average_list

# Save the dictionary to a JSON file
output_file_path = 'data/caltech-101/reference_embeddings.json'
with open(output_file_path, 'w') as json_file:
    json.dump(class_averages, json_file)

print('All class averages have been calculated and saved as JSON.')