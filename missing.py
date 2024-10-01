import os

def find_missing_files(data_dir):
    labels_csv = os.path.join(data_dir, 'labels.csv')
    missing_files = []

    with open(labels_csv, 'r') as f:
        for line in f:
            filename = line.strip().split(',')[0]
            filepath = os.path.join(data_dir, filename)
            if not os.path.isfile(filepath):
                missing_files.append(filename)

    return missing_files

# Specify the directory containing your data
data_directory = 'data/newData'

# Find missing files
missing_files = find_missing_files(data_directory)

# Print missing files
if missing_files:
    print("The following files are listed in labels.csv but are missing from the directory:")
    for filename in missing_files:
        print(filename)
else:
    print("All files listed in labels.csv are present in the directory.")
