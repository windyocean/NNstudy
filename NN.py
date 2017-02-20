import tensorflow as tf
import nipy
from nipy import load_image
import numpy as np
import nibabel as nib

#string : name of imange : size -> 256 * 256 * 256
#find the bound of organ in brain
def find_minmax(string):
    img = nib.load(string)
    img_data = img.get_data()
    X=[256,0]
    Y=[256,0]
    Z=[256,0]
    for x in range(256):
        for y in range(256):
            for z in range(256):
                if img_data[x][y][z]==1:
                    X[0] = min(X[0],x)
                    X[1] = max(X[1],x)
                    Y[0] = min(Y[0],y)
                    Y[1] = max(Y[1],y)
                    Z[0] = min(Z[0],z)
                    Z[1] = max(Z[1],z)
    return X,Y,Z


# in_name : name of input image
# out_name : name of output image

# in_txt : translate input image into in_txt file
# out_txt : translate output image into out_txt file

def change_into_txtfile(in_name, out_name, size_of_box, in_txt, out_txt):
    input_img = nib.load(in_name)
    output_img = nib.load(out_name)
    X, Y, Z = size_of_box[0], size_of_box[1], size_of_box[2]

    while (X[1] - X[0]) % 4 != 0:
        X[1] += 1
    while (Y[1] - Y[0]) % 4 != 0:
        Y[1] += 1
    while (Z[1] - Z[0]) % 4 != 0:
        Z[1] += 1

    input_img = input_img.get_data()[X[0]:X[1], Y[0]:Y[1], Z[0]:Z[1]]
    output_img = output_img.get_data()[X[0]:X[1], Y[0]:Y[1], Z[0]:Z[1]]

    input_img = np.array(input_img).flatten().tolist()
    output_img = np.array(output_img).flatten().tolist()

    with open(in_txt, "w") as f:
        f.write(str(X[0]) + "\n" + str(X[1]) + "\n" + str(Y[0]) + "\n")
        f.write(str(Y[1]) + "\n" + str(Z[0]) + "\n" + str(Z[1]) + "\n")
        for x in input_img:
            f.write(str(x))
            f.write("\n")

    with open(out_txt, "w") as f:
        f.write(str(X[0]) + "\n" + str(X[1]) + "\n" + str(Y[0]) + "\n")
        f.write(str(Y[1]) + "\n" + str(Z[0]) + "\n" + str(Z[1]) + "\n")
        for x in output_img:
            f.write(str(x))
            f.write("\n")

box = find_minmax("Left-Amygdala.nii")
print(box[0],box[1],box[2])

box = [[143,164],[140,158],[142,157]]
change_into_txtfile("T1.nii","Left-Amygdala.nii",box,"in.txt","out.txt")


def weight_variable(shape):
    initial = tf.random_uniform(shape, -0.1, 0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def train_nn_for_organism_detection(list_of_file):
    import copy

    x_collect = []
    y_collect = []

    with open(list_of_file, "r") as f:
        l = f.read().split()
        box_size = copy.deepcopy(l[:6])
        X = list(map(int, [box_size[0], box_size[1]]))
        Y = list(map(int, [box_size[2], box_size[3]]))
        Z = list(map(int, [box_size[4], box_size[5]]))

        for x in range(6, len(l), 2):
            in_txt = l[x]
            out_txt = l[x + 1]

            with open(in_txt, "r") as f:
                l = f.read().split()
                x_collect += list(map(float, l[6:]))
            with open(out_txt, "r") as f:
                l = f.read().split()
                y_collect += list(map(float, l[6:]))

    shape_of_box = [-1, X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0], 1]
    size_of_box = (X[1] - X[0]) * (Y[1] - Y[0]) * (Z[1] - Z[0])

    x = tf.placeholder("float", shape=[None, size_of_box])
    y = tf.placeholder("float", shape=[None, size_of_box])

    ####for train#######

    x_image = tf.reshape(x, shape_of_box)
    y_image = tf.reshape(y, shape_of_box)

    W_conv1 = weight_variable([5, 5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.sigmoid(conv3d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.sigmoid(conv3d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)

    W_fc1 = weight_variable([(size_of_box // 8 // 8) * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, (size_of_box // 8 // 8) * 64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, size_of_box])
    b_fc2 = bias_variable([size_of_box])

    ########end of train#######

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.div(1., 1. + tf.exp(-y_conv))

    result_y = tf.Variable(y_collect)
    # cost = -tf.reduce_mean(result_y*tf.log(tf.clip_by_value(y_conv,1e-4,0.9999))
    #                       + (1-result_y)*tf.log(tf.clip_by_value(1-y_conv,1e-4,0.9999)))
    cost = tf.squared_difference(result_y, y_conv)

    ############################ compare result_y:output and y_conv:from layer

    #############setting optimizer############
    optimizer = tf.train.GradientDescentOptimizer(tf.constant(0.001))
    train = optimizer.minimize(cost)


def TensorboardSetting():
    ## Jaesung : TensorBoard Config
    logs_path = "/tmp/mnist/2"  ###########

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cost)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.merge_all_summaries()


## Jaesung : Split the Session and the function to run the tensorboard
def LetTheGameBegin():
    TensorboardSetting()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    x_collect = np.asarray(x_collect) / 255

    #########################

    print("number of one in result_y", sess.run(result_y).tolist().count(1))

    for epoch in range(81):

        if epoch % 20 == 0:
            print("epoch", epoch)
            tmpy = sess.run(y_conv, feed_dict={x: [x_collect], y: [y_collect], keep_prob: 1})
            cst = sess.run(cost, feed_dict={x: [x_collect], y: [y_collect], keep_prob: 1})
            print(tmpy, sum(sum(cst)))
            # print(yconv[0].tolist().count(1))
            # print(yconv)
            # print(cst)
            # print(sum(elem>0.9 for elem in cst[0]))
        if epoch % 40 == 0:
            res = sess.run(y_conv, feed_dict={x: [x_collect], y: [y_collect], keep_prob: 1})
            num_of_one = sum(1 if x >= 0.5 else 0 for x in res[0])
            print(num_of_one)

        _, summary = sess.run([train, summary_op], feed_dict={x: [x_collect], y: [y_collect], keep_prob: 0.5})

        # write log
        writer.add_summary(summary, epoch)

    with open("yconvyconv.txt", "w") as f:
        f.write(str(num_of_one))
        for x in res[0]:
            f.write(str(x))
            f.write("\n")

