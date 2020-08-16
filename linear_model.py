import itertools
import random
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from absl import flags

import dataset
from vocabulary import Vocabulary

FLAGS = flags.FLAGS
# TODO: what value did Wang et al. use?
flags.DEFINE_float('linear_lambda', 0.001, 'l1 penalty lambda')
flags.DEFINE_integer('bilinear_embedding_dim', 100, 'embedding dimension')
flags.DEFINE_integer('linear_max_updates', 1, 'max updates')
flags.DEFINE_integer('linear_lexical_feature_hash', 0, 'max updates')
flags.DEFINE_integer('linear_logical_feature_hash', 0, 'max updates')
flags.DEFINE_integer('denotations_max_beam_size', 100, ' ')
flags.DEFINE_float('adagrad_initial_lr', 1e-1, 'Adagrad initial learning rate')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(object):
    def __init__(self):
        self.vocab = Vocabulary()
        self.init_parameters()
        self.training_examples = []

    def init_parameters(self):
        raise NotImplementedError()

    def parameters(self):
        raise NotImplementedError()

    def search(self, command):
        raise NotImplementedError()

    def hashable_state(self, state):
        return tuple(map(tuple, state))

    def denotation_log_marginals(self, state, logical_forms, logits):
        # compute {y: \log p_\theta(y | x, s)}
        log_probs = torch.log_softmax(logits, dim=0)
        by_denotation = defaultdict(list)
        for lf_index, lf in enumerate(logical_forms):
            denotation = lf.denotation(state)
            by_denotation[self.hashable_state(denotation)].append(log_probs[lf_index])
        return {
            end_state: torch.logsumexp(torch.stack(this_logits, dim=0), 0)
            for end_state, this_logits in by_denotation.items()
        }

    def predict(self, state, command):
        # argmax over lfs, not denotations (since argmax over denotations assigns mass to empty stacks; see Wang et al.)
        if FLAGS.verbose:
            print(state)
            print(command)
            print(self.vocab.raw_tokens(command))
        lfs, logits = self.lf_logits(command)
        index = logits.argmax()
        lf = lfs[index]
        next_state = lf.denotation(state)
        if FLAGS.verbose:
            print(lf)
            print(next_state)
            print()
        return next_state

    # -> List[LogicalForm], torch.tensor
    def lf_logits(self, command):
        lfs = []
        activations = []
        for featurized_logical_form, score in self.search(command):
            lfs.append(featurized_logical_form.logical_form)
            activations.append(score)
        return lfs, torch.stack(activations, dim=0)

    def regularizer(self):
        raise NotImplementedError()

    def loss(self, state, command, target):
        lfs, logits = self.lf_logits(command)
        denotation_log_marginals = self.denotation_log_marginals(state, lfs, logits)

        hashable_target = self.hashable_state(target)
        # assert hashable_target in denotation_log_marginals, "{} -> {}: {} | {}".format(state, target, hashable_target,
        #                                                                                denotation_log_marginals.keys())
        if hashable_target in denotation_log_marginals:
            loss = - denotation_log_marginals[hashable_target]
            reg = self.regularizer()
            if FLAGS.verbose:
                print("loss: {:.4f}".format(loss))
                print("regularizer: {:.4f}".format(reg))
            loss += reg
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(device)

        return loss

    def optimizer_step(self):
        random.shuffle(self.training_examples)
        for state, command, target in self.training_examples:
            self.optimizer.zero_grad()
            loss = self.loss(state, command, target)
            loss.backward()
            self.optimizer.step()

    def training_accuracy(self):
        n_correct = 0
        for state, command, target in self.training_examples:
            prediction = self.predict(state, command)
            if prediction == target:
                n_correct += 1
        return n_correct / len(self.training_examples)

    def update(self, state, command, target_output, num_updates=None):
        if num_updates is None:
            num_updates = FLAGS.linear_max_updates
        self.training_examples = []
        self.training_examples.append((state, command, target_output))
        it = range(num_updates)
        if num_updates > 1:
            it = tqdm.tqdm(it, ncols=80)
        for _ in it:
            self.optimizer_step()


class Sparse2DLinear(nn.Module):
    def __init__(self, num_a_features, num_b_features):
        super().__init__()
        self.num_a_features = num_a_features
        self.num_b_features = num_b_features
        self.coefficients = nn.Parameter(
            torch.zeros((self.num_a_features, self.num_b_features))
        )

    def l1_norm(self):
        return self.coefficients.norm(p=1)

    def forward(self, a_indices, b_indices):
        return self.coefficients[a_indices][:,b_indices].sum()


class LinearModel(Model):
    def init_parameters(self):
        if FLAGS.linear_lexical_feature_hash > 0:
            lex_features = FLAGS.linear_lexical_feature_hash
        else:
            lex_features = self.vocab.feature_index.size()
        if FLAGS.linear_logical_feature_hash > 0:
            log_features = FLAGS.linear_logical_feature_hash
        else:
            log_features = dataset.LOGICAL_FORM_FEATURE_INDEX.size()

        self.linear = Sparse2DLinear(
            lex_features,
            log_features,
        ).to(device)
        self.regularizer_lambda = FLAGS.linear_lambda
        self.optimizer = optim.Adagrad(self.parameters(), lr=FLAGS.adagrad_initial_lr)
        # self.optimizer = optim.Adadelta(self.linear.parameters())

    def parameters(self):
        return self.linear.parameters()

    def regularizer(self):
        if self.regularizer_lambda != 0.0:
            return self.regularizer_lambda * self.linear.l1_norm()
        else:
            return 0.0

    def search(self, command):
        command_feature_ids = self.vocab.feature_ids(command, bos_and_eos=False)
        if FLAGS.linear_lexical_feature_hash > 0:
            command_feature_ids = [fid % FLAGS.linear_lexical_feature_hash for fid in command_feature_ids]

        command_feature_ids = torch.LongTensor(command_feature_ids).to(device)

        def scoring_function(featurized_logical_form):
            # featurized_logical_form: FeaturizedLogicalForm
            log_feature_ids = featurized_logical_form.feature_ids
            if FLAGS.linear_logical_feature_hash > 0:
                log_feature_ids = [fid % FLAGS.linear_logical_feature_hash for fid in log_feature_ids]

            log_feature_ids = torch.LongTensor(log_feature_ids).to(device)
            activation = self.linear(command_feature_ids, log_feature_ids)
            return activation

        return dataset.search_over_lfs(scoring_function, FLAGS.denotations_max_beam_size)
        # scored = [(flf, scoring_function(flf)) for flf, _ in dataset.FEATURIZED_LOGICAL_FORMS]
        # return scored


class BilinearEmbedding(nn.Module):
    def __init__(self, num_a_features, num_b_features):
        super().__init__()
        self.num_a_features = num_a_features
        self.num_b_features = num_b_features
        self.a_embeddings = nn.Embedding(num_a_features, FLAGS.bilinear_embedding_dim)
        self.b_embeddings = nn.Embedding(num_b_features, FLAGS.bilinear_embedding_dim)
        # don't really need this (won't use the bias), but use it for the initializer
        self.bilinear = nn.Bilinear(
            FLAGS.bilinear_embedding_dim,
            FLAGS.bilinear_embedding_dim,
            1
        )

    def forward_partial(self, a_feat_ids):
        a_feat_ids = torch.LongTensor(a_feat_ids).to(device)
        # emb_dim
        a_embs = self.a_embeddings(a_feat_ids).sum(0)
        inner = torch.einsum("oxy,x->oy", (self.bilinear.weight, a_embs))

        def forward(b_feat_ids):
            b_feat_ids = torch.LongTensor(b_feat_ids).to(device)
            b_embs = self.b_embeddings(b_feat_ids).sum(0)
            activation = torch.einsum("oy,y->o", (inner, b_embs)).squeeze(0)
            return activation

        return forward


class BilinearEmbeddingModel(Model):
    def init_parameters(self):
        self.bilinear = BilinearEmbedding(
            self.vocab.vocab_index.size(),
            dataset.LOGICAL_FORM_FEATURE_INDEX.size()
        ).to(device)
        self.optimizer = optim.Adam(self.parameters())


    def parameters(self):
        return self.bilinear.parameters()

    def regularizer(self):
        return 0.0

    def search(self, command):
        command_feature_ids = self.vocab.token_ids(command, bos_and_eos=False)
        forward = self.bilinear.forward_partial(command_feature_ids)

        def scoring_function(featurized_logical_form):
            # featurized_logical_form: FeaturizedLogicalForm
            activation = forward(featurized_logical_form.feature_ids)
            return activation

        return dataset.search_over_lfs(scoring_function, FLAGS.denotations_max_beam_size)
