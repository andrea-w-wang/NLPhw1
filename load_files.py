import os
import numpy as np

def txt_files_to_list(dir_path, label):
    '''
    create a list of texts along with labels (1 - pos; 0 - neg)
    Arg:
    dir_path = directory that have pos/neg directory
    label = 'pos' or 'neg'
    '''
    file_path = dir_path + label + '/'
    txt_files = os.listdir(file_path)
    txt_list = []
    for i in range(len(txt_files)):
        file = open(file_path + txt_files[i], 'r') 
        txt_list.append(file.read())
    if label == 'pos': 
        return np.stack([txt_list, np.ones(len(txt_files), dtype = int)], axis = -1)
    else: 
        return np.stack([txt_list, np.zeros(len(txt_files), dtype = int)], axis = -1)