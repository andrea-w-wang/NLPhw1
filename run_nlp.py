import pickle as pk
from imdb_indexer import word2index
import sys
import itertools

data_file = sys.argv[1]
upto2gram = pk.load(open(data_file, "rb"))
train_label = pk.load(open("train_label.pk", "rb"))
test_label = pk.load(open("test_label.pk", "rb"))
val_label = pk.load(open("val_label.pk", "rb"))
# create all train tokens 
train_features = upto2gram['train']
val_features = upto2gram['val']
test_features = upto2gram['test']
PAD_IDX = 0
UNK_IDX = 1
max_vocab_size = sys.argv[2]
all_train_features = list(itertools.chain.from_iterable(train_features))
builder = word2index(PAD_IDX, UNK_IDX)
#build vocab
token2id, id2token = builder.build_vocab(all_train_features, max_vocab_size)
train_data_indices = builder.token2index_dataset(train_features)
val_data_indices = builder.token2index_dataset(val_features)
test_data_indices = builder.token2index_dataset(test_features)


##data loader
import numpy as np
import torch
from torch.utils.data import Dataset

class IMDbDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]

def imdb_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]

MAX_SENTENCE_LENGTH = 200

# create pytorch dataloader
BATCH_SIZE = 32
train_dataset = IMDbDataset(train_data_indices, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_collate_func,
                                           shuffle=True)

val_dataset = IMDbDataset(val_data_indices, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_collate_func,
                                           shuffle=True)

test_dataset = IMDbDataset(test_data_indices, test_label)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_collate_func,
                                           shuffle=False)

###

from bow import BagOfWords, test_model
import tqdm

def training(learning_rate, num_epochs, loss_fn, optim, learning_rate_decay = False):
    emb_dim_list = [100, 200, 250, 300, 350]
    result_list = []
    for emb_dim in emb_dim_list:
        dct = {}
        dct['emb'] = emb_dim
        model = BagOfWords(len(id2token), emb_dim)

        # Criterion and Optimizer
        criterion = loss_fn
        optimizer = optim
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        val_acc_list = []
        for epoch in tqdm.trange(num_epochs):
            for i, (data, lengths, labels) in enumerate(train_loader):
                model.train()
                data_batch, length_batch, label_batch = data, lengths, labels
                optimizer.zero_grad()
                outputs = model(data_batch, length_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                val_acc = test_model(val_loader, model)
                if learning_rate_decay == True:
                    scheduler.step(val_acc)
                else: optimizer.step()
                
            val_acc_list.append(val_acc)
        dct['val_acc'] = val_acc_list
        result_list.append(dct)

    return pd.DataFrame.from_dict(result_list)

###
#adam
learning_rate_list = [0.005, 0.01, 0.03, 0.05, 0.1]
num_epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()  
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

plt.rcParams["figure.figsize"] = [20,10]
for r in range(len(learning_rate_list)):
    learning_rate = learning_rate_list[r]

    l = training(learning_rate, num_epochs, loss_fn, optim, learning_rate_decay = False)
    #plot result
    plt.subplot(2,3,r+1)
    for i in range(len(emb_dim_list)):
        plt.plot(np.arange(len(l['val_acc'][i])), l['val_acc'][i], label = 'emb = %s' %emb_dim_list[i])

    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.xticks(np.arange(1, num_epochs+1, step=1))
    plt.title('adam (learning_rate = %s)' %learning_rate)
    plt.legend()
    

l = training(0.1, num_epochs, loss_fn, optim, learning_rate_decay = True)
plt.subplot(2,3,6)
for i in range(len(emb_dim_list)):
        plt.plot(np.arange(len(l['val_acc'][i])), l['val_acc'][i], label = 'emb = %s' %emb_dim_list[i])

plt.ylabel('validation accuracy')
plt.xlabel('epoch')
plt.xticks(np.arange(1, num_epochs+1, step=1))
plt.title('adam (lr decay)')
plt.legend()
plt.savefig(sys.argv[3])


#sgd
learning_rate_list = [0.005, 0.01, 0.03, 0.05, 0.1]
num_epochs = 10
loss_fn = torch.nn.CrossEntropyLoss()  
optim = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)
#set fig size
plt.rcParams["figure.figsize"] = [20,10]
#start
for r in range(len(learning_rate_list)):
    learning_rate = learning_rate_list[r]

    l = training(learning_rate, num_epochs, loss_fn, optim, learning_rate_decay = False)
    #plot result
    plt.subplot(2,3,r+1)
    for i in range(len(emb_dim_list)):
        plt.plot(np.arange(len(l['val_acc'][i])), l['val_acc'][i], label = 'emb = %s' %emb_dim_list[i])

    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.xticks(np.arange(1, num_epochs+1, step=1))
    plt.title('adam (learning_rate = %s)' %learning_rate)
    plt.legend()
    

l = training(0.1, num_epochs, loss_fn, optim, learning_rate_decay = True)
plt.subplot(2,3,6)
for i in range(len(emb_dim_list)):
        plt.plot(np.arange(len(l['val_acc'][i])), l['val_acc'][i], label = 'emb = %s' %emb_dim_list[i])

plt.ylabel('validation accuracy')
plt.xlabel('epoch')
plt.xticks(np.arange(1, num_epochs+1, step=1))
plt.title('adam (lr decay)')
plt.legend()