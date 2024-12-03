# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:27:24 2024

@author: mxjar
"""

### SOURCES
# Kanga gathas
# Yasna
# Zoro-hymns (hymns of athuravan zarathustra)




from PyPDF2 import PdfReader

reader = PdfReader('C:/Users/mxjar/Documents/zoro_text.pdf')
page = reader.pages[100]
extracted_text = page.extract_text()


import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
from PIL import ImageOps

# Ensure Tesseract is installed and set up
# Replace with your tesseract installation path if needed (for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\mxjar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
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














z_t1_path = "C:/Users/mxjar/Documents/zte_test1.pdf"

# Convert PDF pages to images
pages = convert_from_path(z_t1_path, dpi=300)  # dpi=300 for better quality

# Directory to save images (optional)
image_dir = "pdf_images"
#if not os.path.exists(image_dir):
#    os.makedirs(image_dir)

# Initialize an empty string to store the extracted text
z_t1_text = ""

# Iterate through all the pages and extract text
for page_num, page in enumerate(pages):
    # Save the page as an image file (optional)
    #image_path = f"{image_dir}/page_{page_num + 1}.png"
    #page.save(image_path, 'PNG')
    
    page = ImageOps.expand(page, border=500, fill="white")
    # Use PyTesseract to perform OCR on the image
    custom_config = r'-l eng --psm 6'
    text = pytesseract.image_to_string(page.rotate(270), config=custom_config)

    # Append extracted text from this page to the complete PDF text
    z_t1_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"


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
file_path = r'C:/Users/mxjar/Documents/zoro_text_edited2.txt'  # For Windows
# file_path = '/path/to/directory/myfile.txt'  # For Linux/macOS

# Open the file in write mode and save the content
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(z_t1_text)

# Output the extracted text
print(z_t1_text)

with open(file_path, 'r') as file:
    z_t1_text = file.read()











