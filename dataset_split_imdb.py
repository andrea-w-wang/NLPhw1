import os
import numpy as np
import pickle as pk
from load_files import txt_files_to_list as txt2list

pos_trainval = txt2list('./aclImdb/train/', 'pos')
neg_trainval = txt2list('./aclImdb/train/', 'neg')
pos_test = txt2list('./aclImdb/test/', 'pos')
neg_test = txt2list('./aclImdb/test/', 'neg')

np.random.shuffle(pos_trainval)
np.random.shuffle(neg_trainval)

#split train, val
pos_val = pos_trainval[:2500, :]
pos_train = pos_trainval[2500:, :]
neg_val = neg_trainval[:2500, :]
neg_train = neg_trainval[2500:, :]

#put negative and positive data together and shuffle
val_data = np.vstack([pos_val,neg_val])
train_data = np.vstack([pos_train,neg_train])
test_data = np.vstack([pos_test,neg_test])
np.random.shuffle(val_data)
np.random.shuffle(train_data)
np.random.shuffle(test_data)

#separate label and text
train_text = train_data[:,0]; val_text = val_data[:,0]; test_text = test_data[:,0]
train_label = train_data[:,1].astype(int); val_label = val_data[:,1].astype(int); test_label = test_data[:,1].astype(int)

#save label and text
pk.dump(val_text, open("val_text.pk", "wb"))
pk.dump(train_text, open("train_text.pk", "wb"))
pk.dump(test_text, open("test_text.pk", "wb"))

pk.dump(val_label, open("val_label.pk", "wb"))
pk.dump(train_label, open("train_label.pk", "wb"))
pk.dump(test_label, open("test_label.pk", "wb"))