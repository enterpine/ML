#!/usr/bin/env python


from __future__ import print_function

import utils
from model import SeqGAN

import argparse
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a SeqGAN model on some text.')
    parser.add_argument('-t', '--text', default='midi_statematrixs_flatten_merge.txt', type=str,
                        help='path to the text to use')
    parser.add_argument('-l', '--seq_len', default=156*2, type=int,
                        help='the length of each training sequence')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='size of each training batch')
    parser.add_argument('-n', '--num_steps', default=50, type=int,
                        help='number of steps per epoch')
    parser.add_argument('-e', '--num_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('-c', '--only_cpu', default=False, action='store_true',
                        help='if set, only build weights on cpu')
    parser.add_argument('-p', '--learn_phase', default=5, type=int,
                        help='learning phase (None for synchronized)')
    parser.add_argument('-d', '--logdir', default='model/', type=str,
                        help='where to store the trained model')

    args = parser.parse_args()

    # Turns on logging.
    import logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    dictionary, rev_dict = utils.get_dictionary(args.text)
    num_classes = len(dictionary)

    iterator = utils.tokenize(args.text,
                              dictionary,
                              batch_size=args.batch_size,
                              seq_len=args.seq_len)

    sess = tf.Session()
    model = SeqGAN(sess,
                   num_classes,
                   logdir=args.logdir,
                   learn_phase=args.learn_phase,
                   only_cpu=args.only_cpu)
    model.build()
    model.load(ignore_missing=True)

    f = open("record.txt", 'w')
    results = []
    for epoch in xrange(1, args.num_epochs + 1):
        for step in xrange(1, args.num_steps + 1):
            logging.info('epoch %d, step %d', epoch, step)
            f.write('epoch %d, step %d\n' % (epoch, step))
            model.train_batch(iterator.next())

        # Generates a sample from the model.
        g = model.generate(4680)
        print(utils.detokenize(g, rev_dict))
        f.write('====='*10 + '\n' + utils.detokenize(g, rev_dict) + '\n' + '====='*10 + '\n')
        results.append(utils.detokenize(g, rev_dict))
        np.save("results.npy", results)

        # Saves the model to the logdir.
        model.save()
    f.write("finished\n")
    f.close()
