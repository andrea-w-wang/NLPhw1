import spacy_tokenize as tok
import nltk_tokenize as tok
import pickle as pk

val_text = pk.load(open("val_text.pk", "rb"))
test_text = pk.load(open("test_text.pk", "rb"))
train_text = pk.load(open("train_text.pk", "rb"))

#-- spacy tokenization -- #
print ("Tokenizing val data")
val_data_tokens = tok.tokenize_dataset(val_text)
pk.dump(val_data_tokens, open("val_data_tokens_spacy.pk", "wb"))

print ("Tokenizing test data")
test_data_tokens = tok.tokenize_dataset(test_text)
pk.dump(test_data_tokens, open("test_data_tokens_spacy.pk", "wb"))

print ("Tokenizing train data")
train_data_tokens = tok.tokenize_dataset(train_text)
pk.dump(train_data_tokens, open("train_data_tokens_spacy.pk", "wb"))

#-- nltk tokenization -- #

print ("Tokenizing val data")
val_data_tokens = tok.tokenize_dataset(val_text)
pk.dump(val_data_tokens, open("val_data_tokens_nltk.pk", "wb"))

print ("Tokenizing test data")
test_data_tokens = tok.tokenize_dataset(test_text)
pk.dump(test_data_tokens, open("test_data_tokens_nltk.pk", "wb"))

print ("Tokenizing train data")
train_data_tokens = tok.tokenize_dataset(train_text)
pk.dump(train_data_tokens, open("train_data_tokens_nltk.pk", "wb"))

