import argparse
import tarfile

def extract_files(tar_filename, num_files):
    with tarfile.open(tar_filename, 'r:gz') as tar:
        files = tar.getnames()

        # Extract the first N files
        for file in files[:num_files]:
            tar.extract(file)


def create_tar_gz(source_dir, output_file):
    with tarfile.open(output_file, mode='w:gz') as archive:
        archive.add(source_dir, arcname='')


def main(args):
    # Call the extract_files function
    extract_files(args.tarfile, args.num_files)

"""

python extract_tar_file_to_inspect.py --tarfile ../../2020-09-08-arxiv-extracts-nofallback-until-2007-068.tar.gz --num_files 10
python extract_tar_file_to_inspect.py --tarfile ./documents --create_tar

"""

if __name__ == '__main__':
        # Define command line arguments
    parser = argparse.ArgumentParser(description='Extract first N files from a .tar.gz archive')
    parser.add_argument('--tarfile', type=str, help='Path to the .tar.gz archive')
    parser.add_argument('--num_files', type=int, help='Number of files to extract')
    parser.add_argument('--create_tar', action="store_true")

    # Parse command line arguments
    args = parser.parse_args()

    if args.create_tar:
        create_tar_gz(args.tarfile, args.tarfile + ".tar.gz")
    else:
        main(args)
