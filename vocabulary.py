from collections import Counter

import dataset
import spacy
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH
from spacy.lang.en import English

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer('unk_threshold', 0, 'max updates')

class Index(object):
    def __init__(self):
        self.items = []
        self.items_id_map = {}
        self.frozen = False

    def size(self):
        return len(self.items)

    def index(self, item):
        if item not in self.items_id_map:
            assert not self.frozen
            self.items_id_map[item] = len(self.items)
            self.items.append(item)
        return self.items_id_map[item]

    def item(self, index):
        return self.items[index]

class Vocabulary(object):
    START = "<START>"
    END  = "<END>"
    UNK  = "<UNK>"
    def __init__(self):
        self.vocab = []
        self.vocab_id_map = {}

        self.vocab_index = Index()
        self.feature_index = Index()

        # tokenizer
        special_cases = {Vocabulary.START: [{ORTH: Vocabulary.START}],
                         Vocabulary.END: [{ORTH: Vocabulary.END}]}
        self.tokenizer = Tokenizer(English().vocab, rules=special_cases)

        self.token_count = Counter()

        for session_id in dataset.get_session_ids():
            for (_, language, _) in dataset.get_session_data(session_id):
                tokens = self.raw_tokens(language, unk=False)
                self.token_count.update(tokens)

        for token, count in self.token_count.most_common():
            if count > FLAGS.unk_threshold:
                self.vocab_index.index(token)

        feature_count = Counter()
        for session_id in dataset.get_session_ids():
            for (_, language, _) in dataset.get_session_data(session_id):
                # tokens = self.raw_tokens(language)
                # for token in tokens:
                #     self.vocab_index.index(token)

                features = self.raw_features(language)
                feature_count.update(features)

        for feature, count in feature_count.most_common():
            self.feature_index.index(feature)

        print("vocab index size: {}".format(self.vocab_index.size()))
        print("feature index size: {}".format(self.feature_index.size()))

        self.vocab_index.frozen = True
        self.feature_index.frozen = True

    def raw_tokens(self, language, unk=True):
        # return language.split(' ')
        tokenized = self.tokenizer(language)
        tokens = list(str(t).lower() for t in tokenized)
        if unk:
            tokens = [token if self.token_count[token] > FLAGS.unk_threshold else Vocabulary.UNK
                      for token in tokens]
        return tokens

    def raw_features(self, language):
        # unigrams, bigrams, trigrams, skip-trigrams (from Wang et al.)
        tokens = self.raw_tokens(language)
        for token in tokens:
            yield (token, )
        yield from zip(tokens, tokens[1:])
        yield from zip(tokens, tokens[1:], tokens[2:])
        yield from ((a, None, b) for a, b in zip(tokens, tokens[2:]))

    def feature_ids(self, language):
        return [self.feature_index.index(feat) for feat in self.raw_features(language)]

    def token_ids(self, language, bos_and_eos=True):
        tokens = [self.vocab_index.index(t) for t in self.raw_tokens(language)]
        if not bos_and_eos:
            tokens = tokens[1:-1]
        return tokens

    def get_vocab_size(self):
        return self.vocab_index.size()

