class ProteinTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def build_vocab(self):
        # Default vocabulary for amino acids
        return list("ACDEFGHIKLMNPQRSTVWY")  # Standard amino acids

    def tokenize(self, sequence):
        return [self.token_to_id[aa] for aa in sequence if aa in self.token_to_id]

    def detokenize(self, token_ids):
        return ''.join(self.id_to_token[idx] for idx in token_ids if idx in self.id_to_token)

    def encode(self, sequences):
        return [self.tokenize(seq) for seq in sequences]

    def decode(self, token_ids_list):
        return [self.detokenize(token_ids) for token_ids in token_ids_list]