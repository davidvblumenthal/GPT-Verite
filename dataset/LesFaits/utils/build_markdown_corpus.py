import tarfile
import jsonlines
import markdown
import multiprocessing
from bs4 import BeautifulSoup
import tarfile
import jsonlines
import markdown
import multiprocessing
from tqdm import tqdm
import re
import argparse
from pathlib import Path


import json
import os


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown.markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    html = re.sub(r'{#(.*?)}', ' ', html)
    html = re.sub(r'\[\[@(.*?)]]', ' ', html) 

    html = re.sub(r'\[@(.*?)]', ' ', html)

    html = re.sub(r'\(Figure(.*?)\)', ' ', html)
    html = re.sub(r'\(Table(.*?)\)', ' ', html)
    html = re.sub(r'\(\[(.*?)\)', ' ', html)
    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text
    

def combine_jsonlines(directory_path, combined_filename):
    # Get a list of all the jsonlines files in the directory
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.jsonl')]

    # Open a new file for writing the combined data
    with open(combined_filename, 'w') as combined_file:
        for file_name in file_list:
            # Open each individual file and read its data line by line
            with open(os.path.join(directory_path, file_name)) as input_file:
                for line in input_file:
                    # Write each line of data to the combined file
                    combined_file.write(line)

            # Delete the individual file after it has been read
            os.remove(os.path.join(directory_path, file_name))

    # Return the name of the combined file for convenience
    return combined_filename



def process_file(filepath, output_file, files, files_type):
    with tarfile.open(filepath, 'r:gz') as tar:
        with jsonlines.open(output_file, mode='w') as writer:           
            for member in tqdm(files):
                with tar.extractfile(member) as f:
                    
                    if files_type == "markdown":
                        text = markdown_to_text(f.read().decode('utf-8'))
                    elif files_type == "textfile":

                        text = f.read().decode('utf-8')
                    else:
                        print(f"Filetype {files_type} not implemented!")
                                       
                    data = {'text': text}
                    
                    writer.write(data)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="markdown files to jsonl.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Name of the dataset to load.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save location."
    )

    parser.add_argument(
        "--combine_results",
        action="store_true"
    )
    
    parser.add_argument(
        "--files_type",
        type=str
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        help="Number of processes to use."
    )

    args = parser.parse_args()
    
    #md_archive = "../../../pubmed_central/pubmed_central.tar.gz"
    num_processes = args.num_proc
    
    with tarfile.open(args.dataset_path, 'r:gz') as tar:

        if args.files_type == "markdown":
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.md')]
        elif args.files_type == "textfile":
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.txt')]           

    pool = multiprocessing.Pool(num_processes)
    
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    output_files = [f'{args.output_path}/{i}.jsonl' for i in range(num_processes)]
    
    
    chunksize = len(members) // num_processes
    chunks = [members[i:i+chunksize] for i in range(0, len(members), chunksize)]
    
    args = [(args.dataset_path, output_files[i], chunks[i], args.files_type) for i in range(num_processes)]
    
    pool.starmap(process_file, args)

    if args.combine_results:
        combine_jsonlines(directory_path=args.output_path, combined_filename="combined.jsonl")















