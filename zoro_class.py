# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:27:24 2024

@author: mxjar
"""

### SOURCES
# khordeh avesta -- DONE
# zoro gathas -- DONE
# Kanga gathas - DONE
# vedic gathas - DONE
# Yasna
# Zoro-hymns (hymns of athuravan zarathustra)

# you got a lot of your material for the bible from the university of michigan and 
# https://www.stjohnsarlingtonva.org/Customer-Content/saintjohnsarlington/CMS/files/EFM/Didache_-_Complete_Text.pdf for the didache



from PyPDF2 import PdfReader

reader = PdfReader('C:/Users/mxjar/Documents/zoro_text.pdf')
page = reader.pages[100]
extracted_text = page.extract_text()
poppler_path = "/usr/local/bin" 

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from PIL import ImageOps

# Ensure Tesseract is installed and set up
# Replace with your tesseract installation path if needed (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract'

# Path to the PDF
pdf_path = "C:/Users/mxjar/Documents/zoro_text.pdf"

# Convert PDF pages to images
pages = convert_from_path(pdf_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
pdf_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')

    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng+hin --psm 3'
    text = pytesseract.image_to_string(page, config=custom_config)

    # Append extracted text from this page to the complete PDF text
    pdf_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/zoro_hymns.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(pdf_text)

# Output the extracted text
print(pdf_text)

with open(file_path, 'r') as file:
    pdf_text = file.read()

# now let's trim the contents, first 977 characters were non-religious text
pdf_text = pdf_text[1018:]


import pandas as pd
import numpy as np
import re
import nltk

nltk.download('words')

from nltk.corpus import words
eng_words = {i for i in set(words.words()) if len(i)>1}
eng_words = set(['a'])|eng_words
zoro_words = set(['Mazda',
                  'Ahura',
                  'Gatha',
                  'Gathas',
                  'Rigveda'])
texts = pd.DataFrame({'text':[],'source':[],'rel':[]})


def split_concatenated_words(text):
    """Split concatenated English words with a possible non-English proper noun."""
    religion_words = {'vaishya',
                      'brahmin',
                      'ahura',
                      'mazda',
                      'varuna',
                      'deva',
                      'vedhas',
                      'rudra',
                      'vishnu',
                      'brahma',
                      'yasna',
                      'medhas',
                      'spentas',
                      'spenta',
                      'amesha',
                      'gathas',
                      'gatha',
                      'hormazd'}
    
    if text in religion_words:
        return text
    # Initialize the dynamic programming table
    dp = [None] * (len(text) + 1)
    dp[0] = []  # Base case: an empty string is considered a valid split
    
    # Iterate over the text to find valid splits
    for i in range(1, len(text) + 1):
        for j in range(i):
            left_word = text[j:i]  # Substring from j to i
            if dp[j] is not None and (left_word.lower() in eng_words or proper_noun_heuristic(left_word)):
                dp[i] = dp[j] + [left_word]
                break
    
    # If no valid split was found, try to identify the proper noun
    if dp[-1] is None:
        return handle_proper_noun_case(text, dp)
    return ' '.join(dp[-1])

def proper_noun_heuristic(word):
    """Heuristic to identify possible proper nouns (simple assumption: capitalized and not in dictionary)."""
    return bool(re.match(r'^[A-Z][a-z]+$', word)) and word.lower() not in eng_words

def handle_proper_noun_case(text, dp):
    """Handle cases where one part of the text might be a proper noun."""
    longest_valid_split = None
    
    for i in range(len(text) - 1, 0, -1):
        if dp[i] is not None:
            # Consider the remaining part as a proper noun
            proper_noun = text[i:]
            if proper_noun_heuristic(proper_noun):
                return dp[i] + [proper_noun]
    
    return ''

# Example concatenated word with a proper noun


#now let's start editing (ugh)

#pdf_text = re.match('\(Translation\) :(.+?)\(Word-note\) :',pdf_text)


def is_eng(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

hymns_edited = re.sub('\n','',pdf_text)


matches = re.findall('Translation(.+?)(Word-n|word-n|Remark)',hymns_edited)

results = [i[0] for i in list(matches)]
    

hymn_df = pd.DataFrame({'text':results})

def iterate_until_capital(s):
    # Iterate through the string to find the index of the first capital letter
    for index, char in enumerate(s):
        if char.isupper():  # Check if the character is uppercase
            return s[index:]  # Return the substring from the first capital letter onward
            
    return s

def non_eng(char):
    """Returns True if the character is non-ASCII."""
    return ord(char) > 127  # ASCII range is from 0 to 127

def remove_after_first_non_ascii(s):
    # Iterate through the string to find the first non-ASCII character
    for index, char in enumerate(s):
        if non_eng(char):  # Check if the character is non-ASCII
            return s[:index]  # Return the substring before the non-ASCII character
            
    return s



hymn_df['text2'] = results
hymn_df['text'] = hymn_df['text'].apply(lambda x: remove_after_first_non_ascii(iterate_until_capital(x)))

hymn_df['text'] = hymn_df['text'].str.replace('\n','').str.strip()

text_words = re.sub('[^ a-zA-Z-]|(?<=[a-zA-Z]),(?=[a-zA-Z])','',hymns_edited)
text_words = re.sub('-',' ',text_words)
text_words = re.sub(r'\s+', ' ', text_words)

# now, to split all conjoined words
text_words = ' '.join([split_concatenated_words(i) for i in text_words.split()])
text_words = re.sub(r'\s+', ' ', text_words)

txt_df = pd.DataFrame({'text':text_words.split()})
txt_df = txt_df.groupby(['text']).size().reset_index(name = 'counts')




####### Kanga Gathas interpretation


gathas_path = "C:/Users/mxjar/Documents/zoro_gathas.pdf"

# Convert PDF pages to images
pages = convert_from_path(gathas_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
pdf_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')

    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng+hin --psm 1'
    text = pytesseract.image_to_string(page, config=custom_config)

    # Append extracted text from this page to the complete PDF text
    pdf_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/zoro_gathas.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(pdf_text)

# Output the extracted text
print(pdf_text)

with open(file_path, 'r') as file:
    pdf_text = file.read()

######

k_a_path = "C:/Users/mxjar/Documents/khordeh_avesta_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(k_a_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
k_a_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')

    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page, config=custom_config)

    # Append extracted text from this page to the complete PDF text
    k_a_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/khordeh_avesta_edited.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(k_a_text)

# Output the extracted text
print(k_a_text)

with open(file_path, 'r') as file:
    k_a_text = file.read()

######

s_a_path = "C:/Users/mxjar/Documents/selected_yasna_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(s_a_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
s_a_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')

    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page, config=custom_config)

    # Append extracted text from this page to the complete PDF text
    s_a_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/selected_yasna_edited.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(s_a_text)

# Output the extracted text
print(pdf_text)

with open(file_path, 'r') as file:
    pdf_text = file.read()


######

z_g_e_path = "C:/Users/mxjar/Documents/zoro_gathas_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(z_g_e_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
z_g_e_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')

    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page, config=custom_config)

    # Append extracted text from this page to the complete PDF text
    z_g_e_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/zoro_gathas_edited.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(z_g_e_text)
    
# Output the extracted text
print(z_g_e_text)

with open(file_path, 'r') as file:
    z_g_e_text = file.read()
    
######

z_t_path = "C:/Users/mxjar/Documents/zoro_text_edited_grey.pdf"

# Convert PDF pages to images
pages = convert_from_path(z_t_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
z_t_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'

    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    
    z_t_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/zoro_text_edited.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(z_t_text)

# Output the extracted text
print(z_t_text)

with open(file_path, 'r') as file:
    z_t_text = file.read()

print('hello world')

######

# Define the predefined path to save the file
file_path = r'C:/Users/mxjar/Documents/kanga_yashts_edited.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(pdf_text)

# Output the extracted text
print(pdf_text)

with open(file_path, 'r',encoding="utf8") as file:
    pdf_text = file.read()


## let's start text cleaning the kanga yashts


file_path = r'C:/Users/mxjar/Documents/kanga_yashts.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS


with open(file_path, 'r') as file:
    kanga_yashts = file.read()

eng_words_limited = {i for i in eng_words if len(i)>=3}


ky_texts = re.findall(r'\(\d+\)(.*?)(?=\(\d+\)|$)',pdf_text,re.DOTALL)

kyt_expanded = [re.findall(r'\d+\..*?(?=\d+\.|$)',i,re.DOTALL) for i in ky_texts if len(re.findall(r'\d+\..*?(?=\d+\.|$)',i,re.DOTALL)) > 1]

from nltk.tokenize import sent_tokenize, word_tokenize


tt = ky_texts[52]

ky_cleaned = [re.sub(r"[^a-zA-Z0-9,.(\n)\"']", " ", i) for i in ky_texts]
ky_txt = pd.DataFrame({'text':ky_cleaned})
ky_txt['sent'] = ky_txt['text'].apply(lambda x: sent_tokenize(x))

ky_txt_spaced = pd.DataFrame({'idx':ky_txt['index'], 'sent':ky_txt['sent']}).explode('sent').reset_index(drop=True)
ky_txt_spaced['eng_score'] = ky_txt_spaced['sent'].apply(lambda x: len([i for i in re.sub(r"[^a-zA-Z\"']", " ", x).split() if i in eng_words_limited])/len(x.split()))

ky_txt_spaced['len'] = ky_txt_spaced['sent'].apply(lambda x: len(x))

import matplotlib.pyplot as plt

plt.hist(ky_txt_spaced[ky_txt_spaced['eng_score']>0]['eng_score'], bins=100, color="skyblue", edgecolor="black")

ky_txt_limited = ky_txt_spaced[ky_txt_spaced['eng_score'] >= 0.3].reset_index(drop=True)














ztg_path = "/Users/mattjarvis/Downloads/zoro_text_edited_hc.pdf"

# Convert PDF pages to images
pages = convert_from_path(ztg_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
ztg_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    ztg_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
    
    
    
zg2_path = "/Users/mattjarvis/Downloads/zoro_gathas_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(zg2_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
zg2_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    zg2_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"


# Convert PDF pages to images
pages = convert_from_path(zg2_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)



ka_path = "/Users/mattjarvis/Downloads/khordeh_avesta_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(ka_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality


# Initialize an empty string to store the extracted text
ka_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    ka_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"


# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(ka_text)

ka_text_clean = clean_text(ka_text)
ka_c_list = sent_tokenize(ka_text_clean)

ka_df = pd.DataFrame({'text':ka_c_list,'type':['khordeh_avesta']*len(ka_c_list)})

ka_df.to_csv('/Users/mattjarvis/Documents/ka_df.csv')




ky_path = "/Users/mattjarvis/Downloads/khordeh_avesta_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(ky_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality


# Initialize an empty string to store the extracted text
ky_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    ky_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"


# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(ky_text)

ky_text_cleaned = clean_text(ky_text)
ky_c_list = sent_tokenize(ky_text_cleaned)

ky_df = pd.DataFrame({'text':ky_c_list,'type':['kanga_yashts']*len(ky_c_list)})

ky_df.to_csv('/Users/mattjarvis/Documents/ky_df.csv')


sy_path = "/Users/mattjarvis/Downloads/selected_yasna_edited.pdf"

# Convert PDF pages to images
pages = convert_from_path(sy_path, dpi=300, poppler_path=poppler_path)  # dpi=300 for better quality


# Initialize an empty string to store the extracted text
sy_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    sy_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"


# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(sy_text)



sy_text_cleaned = re.sub(r"\n"," ",sy_text)
sy_text_cleaned = re.sub(r"--- Page [0-9]+ ---","",sy_text_cleaned)
sy_text_cleaned  = re.sub(r"\s+", " ", sy_text_cleaned).strip()
sy_text_cleaned = re.split(r'[A-Z]{4,}.*?[1-9]{2}\.[0-9]+',sy_text_cleaned)

sy_df = pd.DataFrame({'text':sy_text_cleaned,'type':['selected_yasna']*len(sy_text_cleaned)})

sy_df.to_csv('/Users/mattjarvis/Documents/sy_df.csv')

import cv2
import numpy as np

image = np.array(page)
h, w, _ = image.shape
boxes = pytesseract.image_to_boxes(image)
for b in boxes.splitlines():
    b = b.split()
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(image, (x, h), (w, y), (0, 255, 0), 2)
cv2.imwrite("C:/Users/mxjar/Documents/boxes_scan.jpg", image)




# Define the predefined path to save the file
file_path = r'/Users/mattjarvis/Documents/ztg_text_full_hc.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(ztg_text)








import re
import language_tool_python
from symspellpy import SymSpell, Verbosity
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt_tab')

# Load a language correction tool
tool = language_tool_python.LanguageTool('en-US')

# SymSpell setup for OCR error correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

def clean_text(raw_text):
    """
    Cleans and processes OCR text.
    """
    # Step 1: Normalize text
    normalized_text = re.sub(r"\(i+\)","",raw_text)
    normalized_text = re.sub(r"[{}«»‘’“”()*™®°'>=_%¥\ufeff]", "", normalized_text)
    normalized_text = re.sub(r"[\'+ | \'+]", " ", normalized_text)# Remove unnecessary characters
    normalized_text = re.sub(r"[\"+ | \"+]", " ", normalized_text)# Remove unnecessary characters
    normalized_text = re.sub(r"--- Page [0-9]+ ---","",normalized_text)
    normalized_text = re.sub(r"—","-",normalized_text)
    normalized_text = re.sub(r"-[\n]+","",normalized_text)
    normalized_text = re.sub(r"-"," ",normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()  # Normalize whitespace

    # Step 2: Fix common OCR issues
    #ocr_corrections = {
    #    #"Wh": "When",
    #    "Recti- tude": "Rectitude",
    #    "wal": "walk",
    #    "rerejens": "verejena",
    #    "k-along": "walk along",
    #}
    #for key, value in ocr_corrections.items():
    #    normalized_text = normalized_text.replace(key, value)

    # Step 3: Spell correction using SymSpell
    
    corrected_words = []
    stop_words = {"the", "and", "is", "to", "in", "of", "a", "for"}
    religion_words = {'vaishya',
                      'brahmin',
                      'ahura',
                      'mazda',
                      'varuna',
                      'deva',
                      'vedhas',
                      'rudra',
                      'vishnu',
                      'brahma',
                      'yasna',
                      'medhas',
                      'spentas',
                      'spenta',
                      'amesha',
                      'gathas',
                      'gatha',
                      'hormazd'}
    topo_names = {'bactria',
                  'media',
                  'persia'}
    combo_set = stop_words|religion_words|topo_names|eng_words
    
    words = re.split(' |;|,|\n',normalized_text.lower())
    words = [split_concatenated_words(i) for i in words if i != '']
    
    for word in words:
        if re.sub(r'[^a-zA-Z]', '', word).lower() not in combo_set and proper_noun_heuristic(word)==False:
            
            suggestions = sym_spell.lookup(re.sub(r'[^a-zA-Z]', '', word), Verbosity.CLOSEST, max_edit_distance=2)
            corrected_words.append(suggestions[0].term if suggestions else word)
        else:
            corrected_words.append(word)
    corrected_text = " ".join(corrected_words)

    # Step 4: Reconstruct sentences
    sentences = sent_tokenize(corrected_text)

    # Step 5: Grammar and style correction
    cleaned_sentences = []
    for sentence in sentences:
        corrected_sentence = tool.correct(sentence)
        cleaned_sentences.append(corrected_sentence)
    cleaned_text = " ".join(sentences)

    return cleaned_text

ztg_token = sent_tokenize(ztg_text)


# Example Usage
ocr_text = """
All the worlds know Him; and they give to Varuna,
the name, “Vedhas” (Mazda).
Wh
(in India) Vishnu approached Indra, for the
sake of communion—the greater one assimilated the great
one—Vedhas (Mazda) won over the Aryans of Trisadha
( Bactria, Media and Persia ) and led the devotee to Recti-
tude.
"""

cleaned_output = clean_text(ocr_text)
print("Cleaned Text:\n", cleaned_output)


words = re.split(' |;|,|\n',ocr_text)















import cv2
import fitz
import numpy as np

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import io

def enhance_pdf_contrast(input_pdf, output_pdf, contrast_factor=1.5):
    # Open the PDF file
    pdf_document = fitz.open(input_pdf)

    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)  # Get all images on the page

        for img_index, img in enumerate(images):
            xref = img[0]  # Image reference number
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]  # Extract the image bytes
            img_ext = base_image["ext"]  # Image file extension (e.g., 'png', 'jpeg')

            # Open the image using PIL
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Enhance contrast using PIL's ImageEnhance
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced_image = enhancer.enhance(contrast_factor)

            # Save the enhanced image back into memory
            img_buffer = io.BytesIO()
            enhanced_image.save(img_buffer, format=img_ext.upper())

            # Replace the original image in the PDF with the enhanced one
            rects = page.get_image_rects(xref)  # Get the positions of the image(s)
            for rect in rects:  # In case the same image is used multiple times
                page.insert_image(rect, stream=img_buffer.getvalue(), keep_proportion=True)

    # Save the updated PDF without removing redactions
    pdf_document.save(output_pdf)


def check_pdf_encryption(file_path):
    try:
        pdf = fitz.open(file_path)
        if pdf.is_encrypted:
            print(f"The PDF '{file_path}' is encrypted.")
            if pdf.permissions & fitz.PDF_PERM_MODIFY:
                print("The PDF allows modifications.")
            else:
                print("The PDF does NOT allow modifications.")
        else:
            print(f"The PDF '{file_path}' is not encrypted.")
    except Exception as e:
        print(f"Error opening PDF: {e}")

# Example Usage
enhance_pdf_contrast("/Users/mattjarvis/Downloads/zte_pg1.pdf", "/Users/mattjarvis/Documents/zte_pg1_1.pdf", contrast_factor=2.0)



enhance_pdf_contrast("/Users/mattjarvis/Downloads/selected_yasna_edited.pdf", "/Users/mattjarvis/Documents/selected_yasna_edited_hc.pdf", contrast_factor=2.0)
enhance_pdf_contrast("/Users/mattjarvis/Downloads/kanga_yashts_edited.pdf", "/Users/mattjarvis/Documents/kanga_yashts_edited_hc.pdf", contrast_factor=2.0)
enhance_pdf_contrast("/Users/mattjarvis/Downloads/khordeh_avesta_edited.pdf", "/Users/mattjarvis/Documents/khordeh_avesta_edited_hc.pdf", contrast_factor=2.0)
enhance_pdf_contrast("/Users/mattjarvis/Downloads/zoro_gathas_edited.pdf", "/Users/mattjarvis/Documents/zoro_gathas_edited_hc.pdf", contrast_factor=2.0)














from pdf2image import convert_from_path

# Path to your PDF
pdf_path = "/Users/mattjarvis/Downloads/zte_pg1.pdf"

# Manually specify the path to Poppler utilities
poppler_path = "/usr/local/bin"  # Adjust to your actual Poppler path

# Convert PDF to images
pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)

# Print the number of pages extracted
print(f"Number of pages: {len(pages)}")




zoro_texts = pd.concat([ka_df,ky_df,ztg_df,zg2_df,sy_df],axis = 0)
zoro_texts.to_csv('/Users/mattjarvis/Documents/combined_zoro_texts.csv')
zoro_texts['len'] = zoro_texts['text'].str.len()
legit_zt = zoro_texts[zoro_texts['text'].str.len()>30].reset_index(drop = True)


####### Now let's do our Christian texts

import pandas as pd
import numpy as np

bs = pd.read_csv('/Users/mattjarvis/Documents/bible_selected.rtf')


import pandas as pd
from striprtf.striprtf import rtf_to_text
import re

def extract_sentences_from_rtf(file_path):
  """
  Extracts sentences from an RTF file, 
  where each sentence is assumed to start with a number in brackets or parentheses followed by a colon.

  Args:
    file_path: Path to the RTF file.

  Returns:
    A list of extracted sentences.
  """
  with open(file_path, 'r', encoding='utf-8') as file:
    rtf_text = file.read()

  # Convert RTF to plain text
  plain_text = rtf_to_text(rtf_text)

  # Extract sentences using regular expression
  sentences = re.findall(r"\[?\d+\]?\s*\((.*?)\)", plain_text, flags=re.DOTALL) 
  return sentences

bs = extract_sentences_from_rtf('/Users/mattjarvis/Documents/revelation.rtf')



def read_rtf_as_string(file_path):
    """
    Reads an RTF file and returns its content as a string.

    Args:
        file_path: Path to the RTF file.

    Returns:
        A string containing the plain text extracted from the RTF file, 
        or an empty string if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            rtf_text = file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                rtf_text = file.read()
        except UnicodeDecodeError:
            print(f"Error: Unable to decode file {file_path}. Please check file encoding.")
            return "" 

    return rtf_to_text(rtf_text)


bs = read_rtf_as_string('/Users/mattjarvis/Documents/bible_selected.rtf')
bs_df = pd.DataFrame({'text':re.split(r'(\(\d+\))', bs)})

bs_df = bs_df[bs_df['text'].str.contains(r'[0-9]+')==False].reset_index(drop=True)
bs_df['text']  = bs_df['text'].str.replace('\n',' ')
bs_df['type'] = 'selected_bible_verses'

rev = read_rtf_as_string('/Users/mattjarvis/Documents/revelation.rtf')

rev_df = pd.DataFrame({'text':re.findall("[\[\(][0-9]*[\]\)].*?(?=\n)",rev)})
rev_df['text'] = rev_df['text'].apply(lambda x: re.sub("[\[\(][0-9]*[\]\)]","",x)).str.strip()
rev_df = rev_df[rev_df['text'].str.len() > 0].reset_index(drop=True)
rev_df['type'] = 'revelation'

lev = read_rtf_as_string('/Users/mattjarvis/Documents/leviticus.rtf')
lev_df = pd.DataFrame({'text':re.findall("[\[\(][0-9]*[\]\)].*?(?=\n)",lev)})
lev_df['text'] = lev_df['text'].apply(lambda x: re.sub("[\[\(][0-9]*[\]\)]","",x)).str.strip()
lev_df = lev_df[lev_df['text'].str.len() > 0].reset_index(drop=True)
lev_df['type'] = 'leviticus'


lev = read_rtf_as_string('/Users/mattjarvis/Documents/leviticus.rtf')
lev_df = pd.DataFrame({'text':re.findall("[\[\(][0-9]*[\]\)].*?(?=\n)",lev)})
lev_df['text'] = lev_df['text'].apply(lambda x: re.sub("[\[\(][0-9]*[\]\)]","",x)).str.strip()
lev_df = lev_df[lev_df['text'].str.len() > 0].reset_index(drop=True)
lev_df['type'] = 'leviticus'

acts = read_rtf_as_string('/Users/mattjarvis/Documents/acts.rtf')
acts_df = pd.DataFrame({'text':re.findall("[\[\(][0-9]*[\]\)].*?(?=\n)",acts)})
acts_df['text'] = acts_df['text'].apply(lambda x: re.sub("[\[\(][0-9]*[\]\)]","",x)).str.strip()
acts_df = acts_df[acts_df['text'].str.len() > 0].reset_index(drop=True)
acts_df['type'] = 'acts'

one_corinthians = read_rtf_as_string('/Users/mattjarvis/Documents/1_corinthians.rtf')
one_corinthians_df = pd.DataFrame({'text':re.findall("[\[\(][0-9]*[\]\)].*?(?=\n)",one_corinthians)})
one_corinthians_df['text'] = one_corinthians_df['text'].apply(lambda x: re.sub("[\[\(][0-9]*[\]\)]","",x)).str.strip()
one_corinthians_df = one_corinthians_df[one_corinthians_df['text'].str.len() > 0].reset_index(drop=True)
one_corinthians_df['type'] = '1_corinthians'

romans = read_rtf_as_string('/Users/mattjarvis/Documents/romans.rtf')
romans_sentences = re.split(r"\\n[0-9]+", repr(romans))
romans_df = pd.DataFrame({'text':romans_sentences})
romans_df['text'] = romans_df['text'].apply(lambda x: re.sub(r'\\n|\\xa0|[*a-z]\(\"https:.*?\)', '', repr(x)).strip())
romans_df['text'] = romans_df['text'].apply(lambda x: re.sub(r'\\', ' ', repr(x)).strip("\""))
romans_df['text'] = romans_df['text'].apply(lambda x: x.strip("'"))
romans_df['text'] = romans_df['text'].apply(lambda x: x.strip())
romans_df['type'] = 'romans'

galatians = read_rtf_as_string('/Users/mattjarvis/Documents/galatians.rtf')
galatians_sentences = re.split(r"\\n[0-9]+", repr(galatians))
galatians_df = pd.DataFrame({'text':galatians_sentences})
galatians_df['text'] = galatians_df['text'].apply(lambda x: re.sub(r'\\n|\\xa0|[*a-z]\(\"https:.*?\)', '', repr(x)).strip())
galatians_df['text'] = galatians_df['text'].apply(lambda x: re.sub(r'\\', ' ', repr(x)).strip("\""))
galatians_df['text'] = galatians_df['text'].apply(lambda x: x.strip("'"))
galatians_df['text'] = galatians_df['text'].apply(lambda x: x.strip())
galatians_df['text'] = galatians_df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))
galatians_df['type'] = 'galatians'

didache = read_rtf_as_string('/Users/mattjarvis/Documents/didache.rtf')
didache_sentences = re.split(r"\\n[0-9]+:[0-9]+", repr(didache))
didache_df = pd.DataFrame({'text':didache_sentences})
didache_df['text'] = didache_df['text'].apply(lambda x: re.sub(r'\\n|\\xa0|[*a-z]\(\"https:.*?\)', '', repr(x)).strip())
didache_df['text'] = didache_df['text'].apply(lambda x: re.sub(r'\\', ' ', repr(x)).strip("\""))
didache_df['text'] = didache_df['text'].apply(lambda x: x.strip("'"))
didache_df['text'] = didache_df['text'].apply(lambda x: x.strip())
didache_df['type'] = 'didache'
didache_df['text'] = didache_df['text'].apply(lambda x: re.sub(r'\\', ' ', repr(x)).strip("\""))
didache_df['text'] = didache_df['text'].apply(lambda x: re.sub('on      \'t','on\'t',x))
didache_df['text'] = didache_df['text'].apply(lambda x: x.strip("'")).apply(lambda x: x.strip()).apply(lambda x: x.strip('"')).apply(lambda x: x.strip()).apply(lambda x: x.strip("'"))
didache_df['text'] = didache_df['text'].apply(lambda x: re.sub(r'\s+\'s', '\'s', x).strip())








