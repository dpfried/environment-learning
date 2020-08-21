import itertools
import pickle
from datetime import datetime
import numpy as np
import sys
import random
import pprint
from collections import defaultdict

import tqdm

from absl import flags
FLAGS = flags.FLAGS

import pretrain
import model as our_model
import baseline_model
import linear_model
import dataset

flags.DEFINE_bool('sandbox', False, 'do nothing (useful for interactive debugging)')
flags.DEFINE_bool('verbose', False, 'print outputs')

flags.DEFINE_enum('training', 'multi', ['multi', 'meta'], 'type of training to use')
flags.DEFINE_integer('multi_epochs', 10, 'number of epochs for multi training')
flags.DEFINE_integer('seed', 1, 'seed for shuffling the training data')
flags.DEFINE_integer('limit_sessions', None, 'max number of sessions')
flags.DEFINE_integer('limit_instances', None, 'max number of instances per session')

def train_multi(model, train_session_ids, val_session_ids):
    train_data = list(
        datum for session_id in train_session_ids
        for datum in dataset.get_session_data(session_id, max_instances=FLAGS.limit_instances)
    )
    rng = random.Random(FLAGS.seed)
    best_model = None
    best_acc = None
    for epoch in range(FLAGS.multi_epochs):
        rng.shuffle(train_data)
        train_stats = defaultdict(list)
        it = tqdm.tqdm(train_data, ncols=80, desc=f'epoch {epoch}')
        for state, language, target_output in it:
            prediction = model.predict(state, language)
            train_stats['is_correct'].append(1 if prediction == target_output else 0)
            train_stats['losses'].append(model.update(state, language, target_output))
            it.set_postfix({'train_acc': np.mean(train_stats['is_correct']), 'loss': np.mean(train_stats['losses'])})
        print('epoch {}\ttrain_overall_acc: {:.4f}\ttrain_loss: {:.4f}'.format(
            epoch, np.mean(train_stats['is_correct']), np.mean(train_stats['losses'])
        ))
        val_stats = test_sessions(model, val_session_ids, name='val')
        # pprint.pprint(val_stats)
        val_overall_acc = np.mean(val_stats['is_correct'])
        val_avg_session_acc = np.mean(val_stats['session_accuracies'])
        print('epoch {}\tval_overall_acc: {:.4f}\tval_avg_session_acc: {:.4f}'.format(
            epoch, val_overall_acc, val_avg_session_acc
        ))
        if best_acc is None or val_overall_acc > best_acc:
            print("new best in epoch {}".format(epoch))
            best_model = pickle.loads(pickle.dumps(model))
            best_acc = val_overall_acc
    return best_model

def train_meta(model, train_session_ids):
    pass

def test_data(model, data):
    stats = defaultdict(list)
    for example_ix, (state, language, target_output) in enumerate(data):
        predicted = model.predict(state, language)
        stats['is_correct'].append(1 if predicted == target_output else 0)
    return stats

def test_sessions(model, test_session_ids, name=''):
    overall_stats = defaultdict(list)
    for session_id in test_session_ids:
        session_stats = test_data(
            model,
            list(dataset.get_session_data(session_id, max_instances=FLAGS.limit_instances))
        )
        if len(session_stats['is_correct']) > 0:
            acc = np.mean(session_stats['is_correct'])
        else:
            acc = 0.0
        for key, value in session_stats.items():
            overall_stats[key] += value
        overall_stats['session_accuracies'].append(acc)

    print('{} number of examples: {}'.format(name, len(overall_stats['is_correct'])))
    print('{} number of correct examples: {}'.format(name, np.sum(overall_stats['is_correct'])))
    print('{} overall accuracy: {:.4f}'.format(name, np.mean(overall_stats['is_correct'])))
    print('{} number of sessions: {}'.format(name, len(overall_stats['session_accuracies'])))
    print('{} mean session accuracy: {:.4f}%'.format(name, np.mean(overall_stats['session_accuracies'])))
    print('{} std session accuracy: {:.4f}%'.format(name, np.std(overall_stats['session_accuracies'])))
    return overall_stats

def evaluate_meta():
    session_ids = list(sorted(dataset.get_session_ids()))
    # don't adjust this seed, for consistency
    rng = random.Random(1)
    rng.shuffle(session_ids)
    if FLAGS.limit_sessions is not None:
        session_ids = session_ids[:FLAGS.limit_sessions]
    N_train = int(len(session_ids) * 0.8)
    N_val = int(len(session_ids) * 0.1)
    N_test = len(session_ids) - N_train - N_val
    train_session_ids = session_ids[:N_train]
    val_session_ids = session_ids[N_train:N_train+N_val]
    test_session_ids = session_ids[-N_test:]
    print(f"{len(train_session_ids)} train sessions")
    print(f"{len(val_session_ids)} val sessions")
    print(f"{len(test_session_ids)} test sessions")
    assert not (set(train_session_ids) & set(test_session_ids)), "overlap between train and test!"
    assert not (set(val_session_ids) & set(test_session_ids)), "overlap between val and test!"
    assert not (set(val_session_ids) & set(train_session_ids)), "overlap between train and val!"

    model = Model()

    model = train_multi(model, train_session_ids, val_session_ids)
    test_stats = test_sessions(model, test_session_ids, name='test')
    # pprint.pprint(test_stats)

if __name__ == '__main__':
    print(sys.argv)
    FLAGS(sys.argv)
    dataset.load()
    if not FLAGS.sandbox:
        Model = linear_model.LinearModel
        evaluate_meta()
