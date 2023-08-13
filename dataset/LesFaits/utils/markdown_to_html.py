import argparse
import markdown



def convert_markdown_to_html(input_file, output_file):
    """
    Convert a Markdown file to an HTML file using a Jinja2 template.
    
    :param input_file: Path to the input Markdown file.
    :type input_file: str
    :param output_file: Path to the output HTML file.
    :type output_file: str
    """
    # Load the Markdown file
    with open(input_file, "r") as md_file:
        md_text = md_file.read()

    # Convert the Markdown to HTML
    html_text = markdown.markdown(md_text)


    # Write the output HTML file
    with open(output_file, "w") as output_file:
        output_file.write(html_text)


"""

python markdown_to_html.py --input ./documents/1409.4236_extract_m.tex.md --output test.html

"""


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert a Markdown file to an HTML file')
    parser.add_argument('--input', metavar='INPUT_FILE', type=str,
                        help='the input Markdown file')
    parser.add_argument('--output', metavar='OUTPUT_FILE', type=str,
                        help='the output HTML file')
    args = parser.parse_args()

    convert_markdown_to_html(args.input, args.output)


