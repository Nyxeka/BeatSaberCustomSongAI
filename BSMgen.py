from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import csv, os

tf.logging.set_verbosity(tf.logging.INFO)

# Application logic...

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    # Input Layer
    input_layer = tf.reshape(
        features["x"],
        [-1,1024] ) # batch_size, beatset length

    """
    batch_size is `set to -1.

    this means that it should be computed, dynamically, on the fly, depending on the number of input values in 'features["x"]'

    This means that we can use batch_size as a 'hyperparameter' that we can tune as we see fit. We can choose how many images we want to feed it at once with a command.

    beatsets length to be 1024

    """

    reshape1 = tf.reshape(input_layer,[32,32])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters=64,
        kernel_size=10,
        padding="same",
        activation=tf.nn.relu)

    
    reshape3 = tf.reshape(conv1,[1024,12])

    #layer 2 we want 1024x12

    denseLayer = tf.layers.dense(inputs=reshape3,
        units=1024,
        activation=tf.nn.relu)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv1d(
        inputs=denseLayer,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size = [2, 2], 
        strides = 2)

    """
    this convolutional layer takes 64 5x5 filters. 64 channels for the 64 filters applied.
    [batch_size,14,14,64]

    the pool will break it down again, this time into a output of [batch_size,7,7,64]
    """

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # change the shape of the output of pool2 to a 3,136 wide set of data.
    denseLayer2 = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu) # send the 3136 wide set of data towards a 1024 set of neurons.
    
    # output layer
    outputLayer = tf.layers.dense(inputs=denseLayer2,
        units=1024,
        activation=tf.nn.relu)


    #====================================================
    #
    #   after this line is all old stuff from the example
    #   code, from the TF tutorial.
    #
    #====================================================

    # Configure the Training Op (for TRAIN modes)
    # Now, we need to optimize our model based on loss. We use a well-known optimization algorithm called 'gradient-descent' TF is so useful!
    # We only do this in training mode, to optimize the model.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # minimize trains the model to get the smallest amount of loss (loss, which we calculated earlier). we input loss, and the number of batches we've gone througg (global step)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)


    # Add evaluation metrics (for EVAL mode)
    # When we run this thing for real, we want to really SEE how well it performed. This gives us metrics on how well it performed.
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,predictions=predictions["classes"]
        )
    }

    # After calculating the metrics, we turn the EstimatorSpec with our mode, loss, and accuracy ratings.
    return tf.estimator.EstimatorSpec(
        mode=mode,loss=loss,eval_metric_ops=eval_metric_ops
    )



# Load training, and Test the data:
def main(args):
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data_list = []
    train_labels_list = []
    for _file in os.listdir("Training-Data"):
        with open('Training-Data/'+ _file[:-5] + '.csv', 'r') as csvFile:
            _r = csv.reader(csvFile)
            i=0
            for row in _r:
                # each row divided up into an array of values.
                # [0] is the beat.
                train_data_list.append(row[0])
                train_labels_list.append([row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12]])
                i += 1

            for i in range(1024-i,1024):
                train_data_list.append(float(0))
                train_labels_list.append([0,0,0,0,0,0,0,0,0,0,0,0])
                
    train_data = np.asarray(train_data_list,dtype=np.float)
    train_labels = np.asarray(train_labels_list,dtype=np.int8)

    eval_data_list = []
    eval_labels_list = []
    for _file in os.listdir("Training-Data/eval"):
        with open('Training-Data/eval/'+ _file[:-5] + '.csv', 'r') as csvFile:
            _r = csv.reader(csvFile)
            i=0
            for row in _r:
                # each row divided up into an array of values.
                # [0] is the beat.
                eval_data_list.append(row[0])
                eval_labels_list.append([row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12]])
                i += 1

            for i in range(1024-i,1024):
                eval_data_list.append(float(0))
                eval_labels_list.append([0,0,0,0,0,0,0,0,0,0,0,0])
                
    eval_data = np.asarray(train_data_list,dtype=np.float)
    eval_labels = np.asarray(train_labels_list,dtype=np.int8)



    """train_data = mnist.train.images # Returns numpy array of the trianing images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns numpy array of the test images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)"""

    # Create the Estimator:
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")

    # Lets watch it happen live with some logging functions as it trains:
    tensors_to_log = {"probabilities": "softmax_tensor"}
    #softmax_tensor is the name we gave our model in cnn_model_fn, and probabilities can be found inside it.
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    #inject a hook that lets us log the probabilities every 50 steps of training.

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn = train_input_fn,
        steps=200,
        hooks=[logging_hook])

    # Now Evaluate the model, and print the results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1, # set epochs to 1 so that the model evaluates the metrics over one epoch of data
        shuffle=False) # iterate through data sequentially
    eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
# Run the app
if __name__ == "__main__":
  tf.app.run()