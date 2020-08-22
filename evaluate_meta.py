import pickle
import random
import sys
from collections import defaultdict
from typing import List

import warnings

import numpy as np
import tqdm
from absl import flags

FLAGS = flags.FLAGS

import torch
import linear_model
import dataset

flags.DEFINE_bool('sandbox', False, 'do nothing (useful for interactive debugging)')
flags.DEFINE_bool('verbose', False, 'print outputs')
flags.DEFINE_bool('update_model_on_each_session', False, 'update model on each test session')

flags.DEFINE_enum('training', 'multi', ['multi', 'multi_unmixed', 'reptile', 'none'], 'type of training to use')
flags.DEFINE_integer('multi_epochs', 10, 'number of epochs for multi training')
flags.DEFINE_integer('seed', 1, 'seed for shuffling the training data')
flags.DEFINE_integer('limit_sessions', None, 'max number of sessions')
flags.DEFINE_integer('limit_instances', None, 'max number of instances per session')
flags.DEFINE_float('reptile_beta', 0.1, 'reptile outer step size')
flags.DEFINE_float('reptile_meta_batch_size', 1, 'number of sessions to use in batched reptile update')
flags.DEFINE_bool('reptile_anneal_beta', False, 'decrease reptile_beta to 0 over the course of multi_epochs')


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


def interpolate_parameters(p_1, p_2, p_2_weight):
    # (1 - p_2_weight) * p_1 + p_2_weight * p_2
    p_1 = dict(p_1)
    p_2 = dict(p_2)
    assert 0 <= p_2_weight <= 1
    w1 = 1 - p_2_weight
    assert p_1.keys() == p_2.keys()
    new_params = dict()
    with torch.no_grad():
        for key in p_1:
            new_params[key] = w1 * p_1[key] + p_2_weight * p_2[key]
    return new_params


def average_parameters(named_params: List[dict]):
    if len(named_params) == 1:
        return dict(named_params[0])
    by_name = defaultdict(list)
    for np in named_params:
        np = dict(np)
        for name, parameters in np.items():
            by_name[name].append(parameters)
    with torch.no_grad():
        return {
            name: torch.stack(params, -1).mean(-1)
            for name, params in by_name.items()
        }


def update_parameters(model_to_update: torch.nn.Module, new_named_parameters):
    p_1 = dict(model_to_update.named_parameters())
    p_2 = dict(new_named_parameters)
    assert p_1.keys() == p_2.keys()
    with torch.no_grad():
        for k in p_1:
            p_1[k].copy_(p_2[k])


def train_unmixed(model, train_session_ids, val_session_ids, updates='multi'):
    rng = random.Random(FLAGS.seed)
    best_model = None
    best_acc = None
    train_session_ids = sorted(train_session_ids)
    assert updates in ['multi', 'reptile']
    for epoch in range(FLAGS.multi_epochs):
        rng.shuffle(train_session_ids)
        train_stats = defaultdict(list)

        if updates == 'reptile':
            if FLAGS.reptile_anneal_beta:
                reptile_beta = np.linspace(FLAGS.reptile_beta, 0, FLAGS.multi_epochs)[epoch]
            else:
                reptile_beta = FLAGS.reptile_beta
            print(f'epoch {epoch}: reptile_beta {reptile_beta}')

        reptile_session_params = []
        for session_ix, session_id in enumerate(tqdm.tqdm(train_session_ids, ncols=80, desc=f'epoch {epoch}')):
            if updates == 'multi':
                # update the model
                session_model = model
            elif updates == 'reptile':
                # update a copy
                session_model = pickle.loads(pickle.dumps(model))

            train_session_data = list(dataset.get_session_data(session_id, max_instances=FLAGS.limit_instances))
            train_session_stats = test_data(session_model, train_session_data, update_model=True)
            reptile_session_params.append(session_model.linear.named_parameters())
            session_acc = np.mean(train_session_stats['is_correct'])
            train_stats['session_accuracies'].append(session_acc)
            train_stats['is_correct'] += train_session_stats['is_correct']
            train_stats['losses'] += train_session_stats['losses']

            if updates == 'reptile' and ((session_ix + 1) % FLAGS.reptile_meta_batch_size == 0 or session_ix == len(train_session_ids) - 1):
                averaged = average_parameters(reptile_session_params)
                # TODO: consider annealing the learning rate
                interpolated = interpolate_parameters(
                    model.linear.named_parameters(),
                    averaged,
                    reptile_beta,
                )
                update_parameters(model.linear, interpolated)
                reptile_session_params = []
        print('epoch {}\ttrain_overall_acc: {:.4f}\ttrain_loss: {:.4f}'.format(
            epoch, np.mean(train_stats['is_correct']), np.mean(train_stats['session_accuracies']),
            np.mean(train_stats['losses'])
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


def test_data(model, data, update_model):
    stats = defaultdict(list)
    for example_ix, (state, language, target_output) in enumerate(data):
        predicted = model.predict(state, language)
        stats['is_correct'].append(1 if predicted == target_output else 0)
        if update_model:
            stats['losses'].append(model.update(state, language, target_output))
    return stats


def test_sessions(model, test_session_ids, name=''):
    overall_stats = defaultdict(list)
    for session_id in test_session_ids:
        session_model = pickle.loads(pickle.dumps(model))
        session_stats = test_data(
            session_model,
            list(dataset.get_session_data(session_id, max_instances=FLAGS.limit_instances)),
            FLAGS.update_model_on_each_session,
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
    print('{} mean session accuracy: {:.4f}'.format(name, np.mean(overall_stats['session_accuracies'])))
    print('{} std session accuracy: {:.4f}'.format(name, np.std(overall_stats['session_accuracies'])))
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
    val_session_ids = session_ids[N_train:N_train + N_val]
    test_session_ids = session_ids[-N_test:]
    print(f"{len(train_session_ids)} train sessions")
    print(f"{len(val_session_ids)} val sessions")
    print(f"{len(test_session_ids)} test sessions")
    assert not (set(train_session_ids) & set(test_session_ids)), "overlap between train and test!"
    assert not (set(val_session_ids) & set(test_session_ids)), "overlap between val and test!"
    assert not (set(val_session_ids) & set(train_session_ids)), "overlap between train and val!"

    model = Model()

    if FLAGS.training == 'multi':
        model = train_multi(model, train_session_ids, val_session_ids)
    elif FLAGS.training == 'multi_unmixed':
        model = train_unmixed(model, train_session_ids, val_session_ids, updates='multi')
    elif FLAGS.training == 'reptile':
        # reptile does update on each session; ensure training matches test
        assert FLAGS.update_model_on_each_session
        model = train_unmixed(model, train_session_ids, val_session_ids, updates='reptile')
    elif FLAGS.training == 'none':
        pass
        val_stats = test_sessions(model, val_session_ids, name='val')
    test_stats = test_sessions(model, test_session_ids, name='test')
    # pprint.pprint(test_stats)


if __name__ == '__main__':
    print(sys.argv)
    FLAGS(sys.argv)
    dataset.load()
    Model = linear_model.LinearModel
    if not FLAGS.sandbox:
        evaluate_meta()
