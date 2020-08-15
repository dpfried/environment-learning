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
flags.DEFINE_integer('linear_max_updates', 50, 'max updates')

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
        self.linear = Sparse2DLinear(
            self.vocab.feature_index.size(),
            dataset.LOGICAL_FORM_FEATURE_INDEX.size()
        ).to(device)

        all_params = list(self.linear.parameters())
        self.optimizer = optim.Adagrad(all_params)
        self.training_examples = []

        self.regularizer_lambda = FLAGS.linear_lambda

    # -> List[LogicalForm, Counter[(vocab_feature_id: int, lf_feature_id: int)]
    def cross_featurized_lfs(self, command):
        command_feature_ids = self.vocab.feature_ids(command)
        cross_featurized_lfs = []
        for featurized_lf in dataset.FEATURIZED_LOGICAL_FORMS:
            crossed_features = Counter(itertools.product(command_feature_ids, featurized_lf.feature_ids))
            cross_featurized_lfs.append((featurized_lf.logical_form, crossed_features))
        return cross_featurized_lfs

    # -> List[LogicalForm], torch.tensor
    def lf_logits(self, command):
        lfs = []
        activations = []
        for logical_form, cross_features in self.cross_featurized_lfs(command):
            lfs.append(logical_form)
            activations.append(self.linear(cross_features))
        # flatten's
        return lfs, torch.stack(activations, dim=0)

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
        lfs, logits = self.lf_logits(command)
        index = logits.argmax()
        lf = lfs[index]
        return lf.denotation(state)

    def optimizer_step(self):
        random.shuffle(self.training_examples)
        for state, command, target in tqdm.tqdm(self.training_examples, ncols=80):
            print(command)
            self.optimizer.zero_grad()

            lfs, logits = self.lf_logits(command)
            denotation_log_marginals = self.denotation_log_marginals(state, lfs, logits)

            hashable_target = self.hashable_state(target)
            assert hashable_target in denotation_log_marginals, (hashable_target, denotation_log_marginals.keys())

            loss = - denotation_log_marginals[hashable_target]
            loss += self.regularizer_lambda * self.linear.l1_norm()

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
        self.training_examples.append((state, command, target_output))
        for _ in range(num_updates):
            self.optimizer_step()

