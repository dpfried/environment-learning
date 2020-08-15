import numpy as np
import random
import itertools

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dataset
from vocabulary import Vocabulary
import message_flags
import discrete_util

from collections import Counter, defaultdict

from absl import flags
FLAGS = flags.FLAGS
# TODO: what value did Wang et al. use?
flags.DEFINE_float('linear_lambda', 0.01, 'l1 penalty lambda')
flags.DEFINE_integer('linear_max_updates', 1, 'max updates')
flags.DEFINE_integer('denotations_max_beam_size', 100, ' ')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Sparse2DLinear(nn.Module):
    def __init__(self, num_a_features, num_b_features):
        super().__init__()
        self.num_a_features = num_a_features
        self.num_b_features = num_b_features
        self.coefficients = nn.Parameter(
            torch.zeros(
                self.num_a_features, self.num_b_features
            )
        )

    def l1_norm(self):
        return self.coefficients.flatten().norm(p=1)

    def forward(self, feat_dict):
        activation = torch.tensor(0.0, requires_grad=True)
        for (a_index, b_index), value in feat_dict.items():
            assert 0 <= a_index < self.num_a_features
            assert 0 <= b_index < self.num_b_features
            activation = activation + self.coefficients[a_index, b_index] * value
        return activation

class Model(object):
    def __init__(self):
        self.vocab = Vocabulary()
        self.init_parameters()
        self.training_examples = []

    def init_parameters(self):
        raise NotImplementedError()

    # -> List[LogicalForm], torch.tensor
    def lf_logits(self, command):
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
        lfs, logits = self.lf_logits(command)
        index = logits.argmax()
        lf = lfs[index]
        next_state = lf.denotation(state)
        if FLAGS.verbose:
            print(lf)
            print(next_state)
            print()
        return next_state

    def loss(self, state, command, target):
        lfs, logits = self.lf_logits(command)
        denotation_log_marginals = self.denotation_log_marginals(state, lfs, logits)

        hashable_target = self.hashable_state(target)
        assert hashable_target in denotation_log_marginals, "{} -> {}: {} | {}".format(state, target, hashable_target, denotation_log_marginals.keys())

        loss = - denotation_log_marginals[hashable_target]
        loss += self.regularizer_lambda * self.linear.l1_norm()
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
        for _ in range(num_updates):
            self.optimizer_step()

class LinearModel(Model):
    def __init__(self):
        self.vocab = Vocabulary()
        self.linear = Sparse2DLinear(
            self.vocab.feature_index.size(),
            dataset.LOGICAL_FORM_FEATURE_INDEX.size()
        ).to(device)

        all_params = list(self.linear.parameters())
        self.optimizer = optim.Adagrad(all_params)
        self.training_examples = []

        self.regularizer_lambda = FLAGS.linear_lambda

    def init_parameters(self):
        self.linear = Sparse2DLinear(
            self.vocab.feature_index.size(),
            dataset.LOGICAL_FORM_FEATURE_INDEX.size()
        ).to(device)
        self.regularizer_lambda = FLAGS.linear_lambda
        self.optimizer = optim.Adagrad(self.linear.parameters())

    def search(self, command):
        command_feature_ids = self.vocab.feature_ids(command)
        def scoring_function(featurized_logical_form):
            # featurized_logical_form: FeaturizedLogicalForm
            cross_features = Counter(itertools.product(command_feature_ids, featurized_logical_form.feature_ids))
            activation = self.linear(cross_features)
            return activation
        return dataset.search_over_lfs(scoring_function, FLAGS.denotations_max_beam_size)

    # -> List[LogicalForm], torch.tensor
    def lf_logits(self, command):
        lfs = []
        activations = []
        for featurized_logical_form, score in self.search(command):
            lfs.append(featurized_logical_form.logical_form)
            activations.append(score)
        # flatten's
        return lfs, torch.stack(activations, dim=0)
