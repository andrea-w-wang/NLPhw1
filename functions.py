
def make_ngram(n, dataset):
    '''
    Args: 
    n: (int) for ngram
    dataset: list with len(# of sentences). Each element is a list of tokenized sentences. 
    
    Return:
    A list with len(# of sentences). Each element is a list of tokenized n-gram. 
    
    '''
    

    ngram = []

    for s in range(len(dataset)):
        ngram.append([" ".join(dataset[s][i+position] for position in range(n)) for i in range(len(dataset[s])-n+1)])
        
    return ngram

def uptoNgram(n, dct_of_ngram):
    '''
    create a dictionary with keys 'train', 'test', 'val'
    where items are a nested list where each element is a list of 1-to-ngram of the original sentence.

    '''
    dct = {}

    for dataset in ['train', 'val', 'test']:
        
        #initialize as unigram
        L = dct_of_ngram[dataset+'_1'][:]

        #add ngram features by sentences
        for i in range(len(L)):
            for x in range(2,n+1):
                L[i] = L[i] + dct_of_ngram['%s_%s'%(dataset, x)][i]
        dct[dataset] = L
    return dct
