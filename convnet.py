import tensorflow as tf
import math



IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
IMAGE_PIXELS = IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS
BATCH_SIZE = 250
DTYPE = tf.float16



def inference():
	images_ph = tf.placeholder(DTYPE, shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
	labels_ph = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

	with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                shape=[5, 5, 3, 32],
                stddev=(1.0 / math.sqrt(IMAGE_PIXELS))
                dtype=DTYPE
            ),
            name='kernel'
        )
        _biases = tf.Variable(tf.constant(0.0, [32], dtype=DTYPE), name='biases')
        _conv = tf.nn.conv2d(images_ph, kernel, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(_conv, _biases), name=scope.name)

    pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool1')

    norm1 = tf.nn.local_response_normalization(pool1, depth_radius=4, alpha=0.0001, beta=0.75, name='norm1')

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                shape=[4, 4, 32, 32],
                stddev=(1.0 / math.sqrt(IMAGE_PIXELS / 4))
                dtype=DTYPE
            ),
            name='kernel'
        )
        _biases = tf.Variable(tf.constant(0.0, [32], dtype=DTYPE), name='biases')
        _conv = tf.nn.conv2d(images_ph, kernel, [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(_conv, _biases), name=scope.name)

    norm2 = tf.nn.local_response_normalization(conv2, depth_radius=4, alpha=0.0001, beta=0.75, name='norm2')

    pool2 = tf.nn.max_pool(norm2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool2')


    return pool2

def loss():
	pass

def training():
	pass
