import os
from collections import defaultdict
import sys
from docx2pdf import convert
import docxpy
from fpdf import FPDF
import regex as re


root_dir = os.getcwd()
sys.path.append(root_dir)

def check_pdf_format(folder_dir):
    '''Checks whether all files in the directory have a pdf version'''

    # Creating a dictionary with extensions and filename_list as key values pairs
    format_dict = defaultdict(lambda: [])
    for f in os.listdir(folder_dir):
        filename, ext = os.path.splitext(f)
        if ext and not filename.startswith('~$'):
            format_dict[ext].append(filename)
    
    # collecting all files that do not have a pdf extension
    not_pdfs = []
    for key in list(format_dict.keys()):
        if key != '.pdf':
            for filename in format_dict[key]:
                if filename not in format_dict['.pdf']:
                    not_pdfs.append(filename + key)
    
    if not_pdfs:
        print(f'The following files are not in .pdf format: {str(not_pdfs)}')
        return False
    else:
        print(f'All files in the directory have a pdf version')
        return True

def check_pdf_exists(folder_dir:str, file:str):
    """
    Check whether pdf exists, returns True if exist.
    Args:
        folder_dir(str): directory path
        file: XXX.pdf  or XXX.doc 
    """
    pdf_name = file.split('.')[0] + '.pdf'
    if os.path.exists(os.path.join(folder_dir, pdf_name)):
        return True

def doc2pdf(folder_dir):
    '''Converts all word documents into pdfs (saves to same directory)'''
    # selecting all .doc and .docx files
    doc_dirs = []
    for f in os.listdir(folder_dir):
        if (f.endswith(".doc") or f.endswith(".docx")) and not f.startswith('~$r'):
            # If not already converted to pdf, append to doc_dirs
            if not check_pdf_exists(folder_dir, f):
                doc_dirs.append(os.path.join(folder_dir, f))
        
        if doc_dirs is not None:
            for path in doc_dirs: #one Doc file.

                try:
                    pdf = FPDF()

                    filepath = os.path.splitext(path)[0]
                    out_file = filepath + '.pdf'

                    text = docxpy.process(path)    # poller = document_analysis_client.begin_analyze_document("prebuilt-layout", document)
                    list_text = text.replace('\t','').replace('/', '').replace("–", '').replace("\u2019", '').replace("\u0116",'').replace('\u201d','').replace('\u201c','').split("\n")
                    list_text = [ c.strip() for c in list_text if '' != c]

                    # Add a page
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size = 12)

                    for i in range(len(list_text)):
                        try:
                            pdf.cell(200, 12, txt = list_text[i], ln =1)
                        except:
                            pass
                    # Saving to pdf.
                    pdf.output(out_file)

                except Exception as e:
                    pass
        else:
            print ('nothing to convert into pdf.')
