import spacy
import string



#tokenized
def tokenize_dataset(dataset):
    tokenizer = spacy.load('en_core_web_sm')
    punctuations = string.punctuation

    def tokenize(sent):
        tokens = tokenizer(sent)
        return [token.text.lower() for token in tokens if (token.text not in punctuations)]
    
    token_dataset = []
    
    for sample in dataset:
        tokens = tokenize(str(sample))
        token_dataset.append(tokens)
        
    return token_dataset

