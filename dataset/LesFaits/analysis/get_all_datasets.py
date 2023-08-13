import os
import argparse
import json

def get_files_recursive(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


"""

python get_all_datasets.py --directory ../../staging_area/

"""


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Get file paths from a directory and its subdirectories')

    # Add the directory argument
    parser.add_argument('--directory', type=str, help='The path to the directory to search')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided directory path
    file_paths = get_files_recursive(args.directory)

    # Create a list to store the file paths as dictionaries
    file_data = []
    for file_path in file_paths:
        # Get the file name, its path, and its type
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        path = os.path.abspath(file_path)
        file_type = os.path.relpath(os.path.dirname(file_path), args.directory)

        # Create a dictionary with the file data
        data = {
            'name': file_name,
            'path': path,
            'type': file_type
        }

        # Add the dictionary to the list
        file_data.append(data)

    # Save the list to a JSON file
    json_file = args.directory + '.json'
    with open(json_file, 'w') as f:
        json.dump(file_data, f, indent=4)

    # Print confirmation message
    print(f'File paths saved to {json_file}')





