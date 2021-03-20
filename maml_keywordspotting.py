#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import numpy as np
import torch
import learn2learn as l2l
import os
from torch import nn, optim

# import tensorflow as tf
# import datetime
from torch.utils.tensorboard import SummaryWriter


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=10000,
        cuda=True,
        seed=42,
        mode='mel_spec',
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda:
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Load train/validation/test tasksets using the benchmark interface
    # mode = spec, mel_spec, mfcc
    tasksets = l2l.vision.benchmarks.get_tasksets('meta_kws_tasksets',
                                                  train_ways=ways,
                                                  train_samples=2*shots,
                                                  test_ways=ways,
                                                  test_samples=2*shots,
                                                  num_tasks=20000,
                                                  root='/home/daniel094144/sam/meta-KWS/data_t/meta_kws/train',
                                                  mode=mode
    )

    # Create model
    if mode == 'spec':
        model = l2l.vision.models.ResNet12(hidden_size = 5120, output_size=ways)
    if mode == 'mel_spec':
        model = l2l.vision.models.ResNet12(hidden_size = 2560, output_size=ways)
    if mode == 'mfcc':
        model = l2l.vision.models.ResNet12(hidden_size = 5120, output_size=ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # tensorboard preprocess
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    # valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
    # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    writer = SummaryWriter("logs/mel_spec")





    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', meta_train_error / meta_batch_size, step=iteration)
        #     tf.summary.scalar('accuracy', meta_train_accuracy / meta_batch_size, step=iteration)
        # with valid_summary_writer.as_default():
        #     tf.summary.scalar('loss', meta_valid_error / meta_batch_size, step=iteration)
        #     tf.summary.scalar('accuracy', meta_valid_accuracy / meta_batch_size, step=iteration)
        writer.add_scalar('train/loss', meta_train_error / meta_batch_size, iteration)
        writer.add_scalar('train/accuracy', meta_train_accuracy / meta_batch_size, iteration)
        writer.add_scalar('valid/loss', meta_valid_error / meta_batch_size, iteration)
        writer.add_scalar('valid/accuracy', meta_valid_accuracy / meta_batch_size, iteration)
        # print('\n')
        # print('Iteration', iteration)
        # print('Meta Train Error', meta_train_error / meta_batch_size)
        # print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        # print('Meta Valid Error', meta_valid_error / meta_batch_size)
        # print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    # with train_summary_writer.as_default():
    #     tf.summary.scalar('loss', meta_test_error / meta_batch_size, step=iteration)
    #     tf.summary.scalar('accuracy', meta_test_accuracy / meta_batch_size, step=iteration)
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)
    meta_value = ['Meta Test Error = ' + str( meta_test_error / meta_batch_size ) + '\n',
                     'Meta Test Accuracy = ' + str( meta_test_accuracy / meta_batch_size ) + '\n']
    with open("Meta_Test.txt", "a") as meta_test_value:
        # meta_test_value = open("Meta_Test.txt", "a")
        meta_test_value.writelines(meta_value)
        # meta_test_value.close()

if __name__ == '__main__':
    main()
