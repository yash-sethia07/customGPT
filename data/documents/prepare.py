import os
import tiktoken
import numpy as np
import string
import nltk
nltk.download('stopwords')

import docx2txt


data = ""


path = "data\documents\Agreements"
  
# Change the directory
os.chdir(path)


#defining the function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    output= [i for i in text if i not in stopwords]
    return output

def preprocess(text):
    # text = remove_punctuation(text)
    text = text.lower()
    text = remove_stopwords(text)
    return text


def read_text_file(file_path):
    # with open(file_path, encoding='cp437') as f:
        # extract text
    text = docx2txt.process(file_path)
    # print(text)
    # return ""
    # print(f.read())
    return preprocess(text) + "\n\n\n\n"
  
  
# iterate through all file
for file in os.listdir():
    print("File found: ", file)
    # Check whether file is in text format or not
    if file.endswith(".docx"):
        # file_path = f"{path}\{file}"
        file_path = f"{file}"
        # call read text file function
        data += read_text_file(file_path)
        print("The Length of building data = ", len(data))




# with open(input_file_path, 'r') as f:
#     data = f.read()
n = len(data)
print("The length of data = ", n)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
