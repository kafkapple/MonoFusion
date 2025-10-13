import os

# Specify the directory where your images are stored
directory = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/data/images/seq1'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    # Check if it's a file and not a directory
    if os.path.isfile(filepath):
        # Extract the numeric part of the filename
        name_part = os.path.splitext(filename)[0]
        
        # Check if the name is numeric and greater than 100
        if name_part.isdigit() and int(name_part) > 99:
            os.remove(filepath)
            print(f"Deleted {filename}")

print("Script completed.")
