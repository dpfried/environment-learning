from datetime import datetime
import numpy as np
import sys

import tqdm

from absl import flags
FLAGS = flags.FLAGS
#
# def del_all_flags(FLAGS):
#     flags_dict = FLAGS._flags()
#     keys_list = [keys for keys in flags_dict]
#     for keys in keys_list:
#         FLAGS.__delattr__(keys)
#
# del_all_flags(FLAGS)

import pretrain
import model as our_model
import baseline_model
import linear_model
import dataset

flags.DEFINE_bool('baseline', False, 'use baseline model')
flags.DEFINE_bool('linear', False, 'use linear model')
flags.DEFINE_bool('bilinear', False, 'use linear model')
flags.DEFINE_bool('batch', False, 'use batch evaluation (only supported with some datasets)')
flags.DEFINE_bool('batch_increasing', False, 'use batch evaluation with larger and larger data sizes')
flags.DEFINE_string('correctness_log', None, 'file to write log indicating which predictions were correct')
flags.DEFINE_bool('sandbox', False, 'do nothing (useful for interactive debugging)')
flags.DEFINE_bool('verbose', False, 'print outputs')
flags.DEFINE_bool('reset_model', True, 'reset the model for each new person')

flags.DEFINE_string('filter_session', None, 'if passed, only run on this session')

def evaluate():
    total_correct = 0
    total_examples = 0
    training_accuracies = []
    start_time = datetime.now()
    if not FLAGS.reset_model:
        model = Model()
    for session_id in dataset.get_session_ids():
        if FLAGS.filter_session is not None and session_id != FLAGS.filter_session:
            continue
        if FLAGS.reset_model:
            model = Model()
        session_correct = 0
        session_examples = 0
        session_correct_list = []

        session_data = list(dataset.get_session_data(session_id))

        if not FLAGS.verbose:
            session_data = tqdm.tqdm(session_data, ncols=80, desc=session_id)

        for example_ix, (state, language, target_output) in enumerate(session_data):
            acc = session_correct / session_examples if session_examples > 0 else 0
            if FLAGS.verbose:
                print("{}: {} / {}\tacc: {:.4f}".format(
                    session_id, example_ix, len(session_data),
                    acc
                ))
            else:
                session_data.set_postfix({'acc': acc})

            predicted = model.predict(state, language)

            if predicted == target_output:
                session_correct += 1
                session_correct_list.append(1)
            else:
                session_correct_list.append(0)
            session_examples += 1
            
            model.update(state, language, target_output)
            training_accuracies.append(model.training_accuracy())
            # if session_examples > 2:
            #     return

        if FLAGS.correctness_log is not None:
            with open(FLAGS.correctness_log, 'a') as f:
                f.write(' '.join(str(c) for c in session_correct_list) + '\n')

        print("this accuracy: {} {} {}".format(datetime.now()-start_time, session_id, session_correct/session_examples))
        total_correct += session_correct
        total_examples += session_examples

    print('overall accuracy: %s%%' % (100*total_correct/total_examples))
    print('average training accuracy: %s%%' % (100*np.mean(training_accuracies)))

def evaluate_batch(data_size, test_size=500):
    results = []
    for session_id in dataset.get_session_ids():
        model = Model()
        session_data = list(dataset.get_session_data(session_id))
        assert len(session_data) > data_size+test_size
        for state, language, target_output in session_data[:data_size]:
            model.update(state, language, target_output, 0)

        if not (FLAGS.linear):
            for i in range(50):
                model.optimizer_step()

        print(' training accuracy: %s%%' % (100*model.training_accuracy()))

        total_correct = 0
        total_examples = 0
        for state, language, target_output in session_data[-test_size:]:
            predicted = model.predict(state, language)
            if predicted == target_output:
                total_correct += 1
            total_examples += 1

        print(' test accuracy: %s%%' % (100*total_correct/total_examples))
        results.append(total_correct/total_examples)
    print('average test accuracy: %s%%' % (100*np.mean(results)))

def evaluate_batch_increasing():
    for data_size in [5,10,21,46,100,215,464,1000,2154,4641]:
        print('data size:', data_size)
        evaluate_batch(data_size)

if __name__ == '__main__':
    print(sys.argv)
    FLAGS(sys.argv)
    dataset.load()
    print("reset model: {}".format(FLAGS.reset_model))
    if not FLAGS.sandbox:
        if FLAGS.linear:
            Model = linear_model.LinearModel
            assert not FLAGS.baseline
        elif FLAGS.bilinear:
            Model = linear_model.BilinearEmbeddingModel
            assert not FLAGS.baseline
        elif FLAGS.baseline:
            Model = baseline_model.Model
        else:
            if not pretrain.saved_model_exists():
                print('No pretrained model found with prefix "%s"; running pretraining' % FLAGS.pretrain_prefix)
                pretrain.train()
            Model = our_model.Model

        if FLAGS.batch_increasing:
            evaluate_batch_increasing()
        elif FLAGS.batch:
            evaluate_batch(100)
        else:
            evaluate()
