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


def process_file(filepath, output_file, files):
    with tarfile.open(filepath, 'r:gz') as tar:
        with jsonlines.open(output_file, mode='w') as writer:
            for member in tqdm(files):
                with tar.extractfile(member) as f:
                    
                    text = markdown_to_text(f.read().decode('utf-8'))
                                       
                    data = {'text': text}
                    writer.write(data)

if __name__ == '__main__':
    md_archive = "../../../pubmed_central/pubmed_central.tar.gz"
    num_processes = 1
    
    with tarfile.open(md_archive, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith('.md')]

    pool = multiprocessing.Pool(num_processes)
    output_files = [f'../../../pubmed_central/pubmed_central_{i}.jsonl' for i in range(num_processes)]
    chunksize = len(members) // num_processes
    chunks = [members[i:i+chunksize] for i in range(0, len(members), chunksize)]
    
    args = [(md_archive, output_files[i], chunks[i]) for i in range(num_processes)]
    
    pool.starmap(process_file, args)









def create_tar_gz(source_dir, output_file):
    with tarfile.open(output_file, mode='w:gz') as archive:
        archive.add(source_dir, arcname='')





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