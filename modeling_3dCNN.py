import tensorflow as tf
import numpy as np


# learning_rate = tf.placeholder("float")
# learning_rate = tf.Variable(0.001)
## I remember that there is a function named "tf.constant" ... ?

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    with tf.name_scope("CNN") as scope:
        l1a = tf.nn.relu(tf.nn.conv3d(X, w,  # wrong -> l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1, 1], padding='SAME'))
        l1 = tf.nn.max_pool3d(l1a, ksize=[1, 2, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 2, 1], padding='SAME')
        l1 = tf.nn.dropout(l1, p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv3d(l1, w2,  #  wrong -> l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool3d(l2a, ksize=[1, 2, 2, 2, 1],  #  wrong -> l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, p_keep_conv)

        l3a = tf.nn.relu(tf.nn.conv3d(l2, w3,  # wrong ->  l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool3d(l3a, ksize=[1, 2, 2, 2, 1],  #  wrong -> l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 2, 1], padding='SAME')
        l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  #  wrong -> reshape to (?, 2048)
        l3 = tf.nn.dropout(l3, p_keep_conv)

        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        pyx = tf.matmul(l4, w_o)
        return pyx

with tf.name_scope("Input") as scope:
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# Here comes the input data
    trX = trX.reshape(-1, 256, 256, 256, 1)  # 256x256x256x1 input img
    teX = teX.reshape(-1, 256, 256, 256, 1)  # 256x256x256x1 input img

    X = tf.placeholder("float", [None, 256, 256, 256, 100], name="X_input")
    Y = tf.placeholder("float", [None, 256, 256, 256, 100], name="Y_input")

with tf.name_scope("weight") as scope:
    w = init_weights([3, 3, 1, 32])  # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 4 * 4, 625])  # 128 filt 4 * 4 img
    w_o = init_weights([625, 256 * 256 * 256])  # FC 625 inputs, 256 * 256 * 256 outputs (labels)

with tf.name_scope("parameter") as scope:
    # import input_data
    batch_size = tf.Variable(128, name="batch_size")
    test_size = tf.Variable(256, name="test_size")
    learning_rate = tf.Variable(0.001, name="learning_rate")

    p_keep_conv = tf.placeholder("float", name = "p_keep_conv")
    p_keep_hidden = tf.placeholder("float", name = "p_keep_hidden")
    py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

with tf.name_scope("cost & train") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    cost_summ = tf.scalar_summary("cost & train", cost)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1) # what is this ??

# Launch the graph in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    merged = tf.merge_all_summaries()
    # I think there is a problem in the string index right below this command line
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph_def)

    for i in range(500): # does 500 mean epoch?
        # what is the operation of the function <zip> ??
        # batch_size ?? what is it?
            # what is the function of the <range> ??
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))

        for start, end in training_batch: # um .. training_batch??
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

            print sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})

        #what the fuck?
        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        ## are numpy and tensor inter-changable ?
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
        if i % 50 == 0:
            # not in this line -> merged = tf.merge_all_summaries()
            summary = sess.run(merged, feed_dict={X: trX[start:end], Y: trY[start:end],
                                            p_keep_conv: 0.8, p_keep_hidden: 0.5})
            writer.add_summary(summary, i)

    # I think this is wrong.
    tensorboard --logidr=/tmp/mnist_logs
