import tensorflow as tf
import numpy as np
import progressbar
import argparse
import glob
import sys
import os


total = 6000


def parse_args():

    desc = "Basic LSTM for text prediction"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--check_path', type=str, default='./check', help='File path to save checkpoints')

    parser.add_argument('--logs_path', type=str, default='./logs', help='File path of logs for tensorboard')

    parser.add_argument('--n_hidden', type=int, default=512, help='Number of hidden units in a single LSTM   cell')

    parser.add_argument('--n_input', type=int, default=128, help='Number of input words')

    parser.add_argument('--n_stacks', type=int, default=1, help='Number of LSTM stacks')

    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

    return check_args(parser.parse_args())

def check_args(args):


    try:
        f = open(args.text_path,'r')
    except Exception:
        print("no such file found to be read")

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
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


tf.reset_default_graph()


def get_data(ind):
    label_path = '/home/knot/Documents/MTP/data/eeg_label.npy'
    input_path = '/home/knot/Documents/MTP/data/input_eeg.npy'

    mat_input = np.load(input_path)
    mat_label = np.load(label_path)

    total = mat_inout.shape[0]
    X = np.zeros((batch_size, 128, 200))
    Y = np.zeros((batch_size, 40))


    for i in range(8):
        temp = mat_input[ind,:,:].T
        X[i,:,:] = temp
        temp = mat_label[ind,:,:]
        temp = np.squeeze(temp, axis = 0)
        Y[i,:] = temp

    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y


def rnn(inp_, args):

    n_input = args.n_input
    n_hidden = args.n_hidden
    n_stacks = args.n_stacks


    inp_ = tf.reshape(inp_, [-1, n_input])
    inp_ = tf.split(inp_, n_input, 1)

    rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_stacks)])
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, inp_, dtype=tf.float32)

    dense1 = tf.layers.dense(inputs = outputs[-1], units = 512, activation = tf.nn.relu, name = "dense1")
    logits = tf.layers.dense(inputs = dense1, units = 40)
    out = tf.nn.softmax(logits, name="softmax_output")

    return out


def main(args):

    logs_path = args.logs_path
    n_input = args.n_input
    n_hidden = args.n_hidden
    epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    check_path = args.check_path

    x = tf.placeholder(tf.float32, shape = (None, 200, 128))
    y = tf.placeholder(tf.float32, shape = (None, 40))

    pred = rnn(x, args)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
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
            rpoch_loss = 0
            for ind in progressbar.progressbar(range(total)):
                batchx = np.zeros((8, n_input, 200))
                batchy = np.zeros((8, vocab_size))

                batchx[0], batchy[0] = get_data(ind)
                batchx = np.array(batchx)
                batchy = np.array(batchy)

                _, loss, onehot_pred, acc, summary = sess.run([optimizer, cost, pred, accuracy, summary_op], feed_dict={x: batchx, y: batchy})
                writer.add_summary(summary, pt)
                pt+=1
                avg_acc +=  acc
                saver.save(sess, os.path.join(check_path,"/home/knot/Documents/MTP/chackpoints1/model.ckpt"))
                epoch_loss += loss

            avg_acc /= len(X)
            print("Epoch: ",epoch,"loss: ", epoch_loss, "acc: ", avg_acc)







if(__name__=="__main__"):
    args = parse_args()
    if args is None:
        exit()
    main(args)
