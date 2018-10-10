from collections import Counter

class word2index():
    def __init__(self, PAD_IDX, UNK_IDX):
        self.PAD_IDX = PAD_IDX
        self.UNK_IDX = UNK_IDX
     
    def build_vocab(self, all_tokens, max_vocab_size):
        # Returns:
        # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
        # token2id: dictionary where keys represent tokens and corresponding values represent indices
        token_counter = Counter(all_tokens)
        vocab, count = zip(*token_counter.most_common(max_vocab_size))
        id2token = list(vocab)
        token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
        id2token = ['<pad>', '<unk>'] + id2token
        token2id['<pad>'] = self.PAD_IDX 
        token2id['<unk>'] = self.UNK_IDX
        self.id2token = id2token
        self.token2id = token2id
        return self.token2id, self.id2token


    # convert token to id in the dataset
    def token2index_dataset(self, tokens_data):
        indices_data = []
        for tokens in tokens_data:
            index_list = [self.token2id[token] if token in self.token2id else self.UNK_IDX for token in tokens]
            indices_data.append(index_list)
        return indices_data