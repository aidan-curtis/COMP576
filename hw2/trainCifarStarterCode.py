from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    b = tf.Variable(tf.constant(0.1, shape=shape))
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters =
    the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the
    filters, a.k.a. the results after convolution
    '''
    
    # IMPLEMENT YOUR CONV2D HERE
    
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


result_dir = './results/rms_lr=1e-4_momentum=1/'


ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 200

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = 'CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = 'CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()






tf_data = tf.placeholder(tf.float32, [None, 28, 28, 1]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels] 
tf_labels = tf.placeholder(tf.float32, [None, 10]) #tf variable for labels

# --------------------------------------------------
# model
# create your model

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

tf.summary.histogram("Layer_1_Activations", h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

tf.summary.histogram("Layer_2_Activations", h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

tf.summary.histogram("fc1_Activations", h_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

tf.summary.histogram("fc2_Activations", y_conv)

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_labels * tf.log(y_conv), reduction_indices=[1]))
optimizer = tf.train.RMSPropOptimizer(1e-3, momentum=1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

# --------------------------------------------------
# optimization


# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()

test_acc_summary = tf.summary.scalar('test_accuracy', accuracy)
train_acc_summary = tf.summary.scalar('train_accuracy', accuracy)
train_loss_summary = tf.summary.scalar('train_loss', cross_entropy)


# Add the variable initializer Op.
init = tf.initialize_all_variables()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# Run the Op to initialize the variables.
sess.run(init)

batch_xs = np.zeros([batchsize, imsize, imsize, 1])#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass])#setup as [batchsize, the how many classes] 
for i in range(500): # try a small iteration size once it works then continue
    perm = np.arange(ntrain*nclass)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]

    if i%10 == 0:
        weights = W_conv1.eval()

        # summary_str = sess.run(summary_op, feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        # summary_writer.add_summary(summary_str, i)
        # summary_writer.flush()

        test_accuracy_summary = sess.run(test_acc_summary, feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
        train_accuracy_summary = sess.run(train_acc_summary, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        train_loss = sess.run(train_loss_summary, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys,  keep_prob: 1.0})
        summary_writer.add_summary(test_accuracy_summary, i)
        summary_writer.add_summary(train_accuracy_summary, i)
        summary_writer.add_summary(train_loss, i)
        print("step %d, training accuracy %g"%(i, train_accuracy))

    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys,  keep_prob: 0.5}) # dropout only during training





# Visualizing conv1 weights
# with open("conv1.weights.npz", "w") as outfile:
#             np.save(outfile, weights)

# --------------------------------------------------
# test

print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


sess.close()