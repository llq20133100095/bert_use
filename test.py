import tensorflow as tf
import numpy as np

# Reinitializable iterator to switch between Datasets
EPOCHS = 10

# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))

# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
features, labels = iter.get_next()

# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        sess.run([features, labels])
    sess.run(test_init_op) # switch to val dataset
    print(sess.run([features, labels]))