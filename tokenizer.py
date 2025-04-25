import collections
import re

class BPETokenizer:
    def __init__(self, vocab_size, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.tokens = {} # Stores tokens
        self.token_id = {} # Maps tokens to their ID
        self.id_token = {} # Maps IDs to tokens

    def tokenize(self, text):
        # Takes a string of text, splits on whitespace, and returns a list of tokens
        tokens = text.split()
        return tokens

    def train(self, corpus):
        # Tokenize the corpus (a list of strings)
        token_freqs = collections.defaultdict(int) # if token not present, store value as 0
        for text in corpus:
            tokens = self.tokenize(text)
            for token in tokens:
                token_freqs[token] += 1 # increment frequency for each time present
        
        # Initialize vocabulary with unique characters
        vocab = set()
        for token in token_freqs:
            for char in token:
                vocab.add(char)
        
        # Merge most frequent pairs
        while len(self.tokens) < self.vocab_size:
            pair_freqs = collections.defaultdict(int)
            for token, freq in token_freqs.items():
                chars = list(token)
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    pair_freqs[pair] += freq
            
            # Find the most frequent pair
            most_freq_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[most_freq_pair] < self.min_freq:
                break
            
            # Merge the most frequent pair
            vocab.add(most_freq_pair[0] + most_freq_pair[1])
            self.tokens[most_freq_pair[0] + most_freq_pair[1]] = len(self.tokens)
            new_token_freqs = collections.defaultdict(int)
            for token, freq in token_freqs.items():
                new_token = token.replace(most_freq_pair[0] + most_freq_pair[1], most_freq_pair[0] + ' ' + most_freq_pair[1])
                new_token_freqs[new_token] += freq
            token_freqs = new_token_freqs
        
        # Build token_id and id_token mappings
        for token in vocab:
            self.token_id[token] = len(self.token_id)
            self.id_token[len(self.id_token)] = token

    def encode(self, text):
        """
        Takes a string of text as input and returns a list of encoded tokens
        """
        tokens = self.tokenize(text)
        encoded = []
        for token in tokens:
            encoded_token = []
            for char in token:
                if char in self.token_id:
                    encoded_token.append(self.token_id[char])
                else:
                    # Handle unknown characters
                    encoded_token.append(len(self.token_id))  # Assume last id + 1 = <unk> (unknown character)
            encoded.append(encoded_token)
        return encoded

    def decode(self, encoded):
        """"
        Takes a list of encoded tokens as input and returns a string of text
        """
        decoded = []
        for encoded_token in encoded:
            token = ''.join([self.id_token.get(id, '<unk>') for id in encoded_token])
            decoded.append(token)
        return ' '.join(decoded)
    
corpus = ["This is a sample example sentence.", "Hello world!"]
tokenizer = BPETokenizer(vocab_size=256)
tokenizer.train(corpus)
encoded = tokenizer.encode("This is a test sentence")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
                

