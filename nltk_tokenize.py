#tokenized
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


def tokenize_dataset(dataset):
    snowball = SnowballStemmer()
    
    def tokenize(sent):
        tokens = word_tokenize(sent)
        stop_words = set(stopwords.words('english'))
        remove_punc = [porter.stem(token.lower()) for token in tokens if token.isalpha()]
        return  [w for w in remove_punc if not w in stop_words]

    token_dataset = []
    
    for sample in dataset:
        tokens = tokenize(str(sample))
        token_dataset.append(tokens)
        
    return token_dataset

