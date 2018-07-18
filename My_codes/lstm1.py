'''
By Abhishek Pal, undegraduate Major Technical Project at Indian Institute of tenchnology
LSTM model for EEG signal classification
Date: 17/7/2018
'''


import tensorflow as tf
import numpy as np
import progressbar
import argparse
import glob
import sys
import os


tf.reset_default_graph()
Y_ = tf.placeholder(tf.float32, shape = (None, 40))
X_ = tf.placeholder(tf.float32, shape = (None, 200, 128))
total = 6000


def parse_args():

    desc = "Basic LSTM for text prediction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--check_path', type=str, default='./check', help='File path to save checkpoints')

    parser.add_argument('--logs_path', type=str, default='./logs', help='File path of logs for tensorboard')

    parser.add_argument('--n_hidden', type=int, default=256, help='Number of hidden units in a single LSTM   cell')

    parser.add_argument('--n_input', type=int, default=200, help='Number of input words')

    parser.add_argument('--n_classes', type=int, default=40, help='Number of classes')

    parser.add_argument('--n_stacks', type=int, default=1, help='Number of LSTM stacks')

    parser.add_argument('--batch_size', type=int, default=4, help='Number of LSTM stacks')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--n_epochs', type=int, default=1000, help='The number of epochs to run')


    return check_args(parser.parse_args())

def check_args(args):

    try:
        os.mkdir(args.logs_path)
    except Exception:
        pass

    files = glob.glob(args.logs_path+'/*')
    for f in files:
        os.remove(f)

    try:
        os.mkdir(args.check_path)
    except Exception:
        pass

    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    try:
        assert args.n_classes >= 1
    except:
        print('number of classes must be larger than one')

    try:
        assert args.n_stacks >= 1
    except:
        print('number of stacked lstm must be atleast one')


    try:
        assert args.n_input >= 1
    except:
        print('number of words must be grater than or equal to one')

    try:
        assert args.learning_rate > 0
    except:
        print('learning rate must be positive')

    try:
        assert args.n_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args




def get_data(ind, args):
    label_path = '/home/knot/Documents/MTP/data/eeg_label.npy'
    input_path = '/home/knot/Documents/MTP/data/input_eeg.npy'
    n_input = args.n_input
    batch_size  = args.batch_size
    n_classes = args.n_classes


    mat_input = np.load(input_path)
    mat_label = np.load(label_path)

    total = mat_input.shape[0]
    X = np.zeros((batch_size, n_input, 128))
    Y = np.zeros((batch_size, n_classes))


    for i in range(batch_size   ):
        temp = mat_input[ind+i,:,:]
        X[i,:,:] = temp
        temp = mat_label[ind+i,:,:]
        temp = np.squeeze(temp, axis = 0)
        Y[i,:] = temp

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def rnn(inp_, args):

    n_input = args.n_input
    n_hidden = args.n_hidden
    n_stacks = args.n_stacks
    n_classes = args.n_classes

    inp_ = tf.unstack(inp_ ,n_input, 1)


    rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_stacks)])
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inp_, dtype=tf.float32)

    dense1 = tf.layers.dense(inputs = outputs[-1], units = 512, activation = tf.nn.relu, name = "dense1")
    logit = tf.layers.dense(inputs = dense1, units = n_classes)
    out = tf.nn.softmax(logit, name="softmax_output")

    return out


def main(args):

    logs_path = args.logs_path
    n_input = args.n_input
    n_hidden = args.n_hidden
    epochs = args.n_epochs
    n_classes = args.n_classes
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    check_path = args.check_path


    pred = rnn(X_, args)
    # print(eval("pred"))
    # print(eval("Y_"))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels=Y_))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy ", accuracy)

    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, os.path.join(check_path,"/home/knot/Documents/MTP/chackpoints1/model.ckpt"))
        '''Training'''

        pt = 0
        for epoch in range(epochs):
            epoch_loss = 0
            avg_acc = 0
            for ind in progressbar.progressbar(range(0,total,batch_size)):
                batchx = np.zeros((batch_size, n_input, 128))
                batchy = np.zeros((batch_size, n_classes))

                batchx, batchy = get_data(ind, args)
                batchx = np.array(batchx)
                batchy = np.array(batchy)
                _, loss, onehot_pred, acc, summary = sess.run([optimizer, cost, pred, accuracy, summary_op], feed_dict={X_: batchx, Y_: batchy})
                writer.add_summary(summary, pt)
                pt+=1
                avg_acc +=  acc
                saver.save(sess, os.path.join(check_path,"/home/knot/Documents/MTP/chackpoints1/model.ckpt"))
                epoch_loss += loss

            avg_acc /= total
            avg_acc *= batch_size
            print("Epoch: ",epoch,"loss: ", epoch_loss, "acc: ", avg_acc)







if(__name__=="__main__"):
    args = parse_args()
    if args is None:
        exit()
    main(args)
