import tensorflow as tf
import numpy as np

import tensorflow.contrib as tf_contrib

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, seq_length, num_classes, vocab_size,
                  embed_size, filter_sizes, num_filters, l2_reg_lambda=0.0, k=3,
                  decay_steps=1000, decay_rate=0.9, clip_gradients=5.0, learning_rate=1e-4):

        self.seq_length  = seq_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.clip_gradients = clip_gradients
        self.learning_rate = learning_rate

        
        self.global_step = tf.get_variable(name='Global_Step', initializer=0, trainable=False)


        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1.0, 1.0), name="W")
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)
            

        features = self.kmax_cnn_layer(self.embedded_words, self.filter_sizes, self.num_filters, k)

        # Add dropout
        with tf.name_scope("dropout"):
            features = tf.nn.dropout(features, self.dropout_keep_prob)

        projection_hidden_size = features.shape[-1].value
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                name="W",
                shape=[projection_hidden_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='b', initializer=tf.constant(0.1, shape=[num_classes]))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(features, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.train_op = self.get_train_op()


    def kmax_cnn_layer(self, embedded_words, filter_sizes, num_filters, k=3):
        embedded_words_expanded = tf.expand_dims(embedded_words, -1)
        embed_size = embedded_words.shape[-1].value
        seq_length = embedded_words.shape[1].value

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embed_size, 1, num_filter]
                W = tf.get_variable(name='W', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable(name='b', initializer=tf.constant(0.1, shape=[num_filter]))
                conv = tf.nn.conv2d(
                    embedded_words_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                '''
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                '''
                #k-max pooling 
                h_ = tf.transpose(h, [0, 3, 2, 1])
                values = tf.nn.top_k(h_, k, sorted=False).values
                pooled = tf.transpose(values, [0, 3, 2, 1])
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = sum(num_filters) * k
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat

    def get_train_op(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_ = learning_rate
        #noise_std_dev = tf.constant(0.3) / (tf.sqrt(tf.cast(tf.constant(1) + self.global_step, tf.float32))) #gradient_noise_scale=noise_std_dev
        train_op = tf_contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                   learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op


if __name__ == '__main__':
    batch_size = 8
    seq_length = 30
    dropout_keep_prob = 0.5
    model = TextCNN(seq_length=seq_length, num_classes=3, vocab_size=2000, embed_size=200, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], l2_reg_lambda=0.1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            # input_x should be:[batch_size, num_sentences,self.seq_len]
            #input_x = np.zeros((batch_size, seq_length+i)) #num_sentences, no padding test
            input_x = np.zeros((batch_size, seq_length)) #num_sentences
            input_x[input_x > 0.5] = 1
            input_x[input_x <= 0.5] = 0
            print(np.shape(input_x))
            input_y = np.matrix(
                [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])  # np.zeros((batch_size),dtype=np.int32) #[None, self.seq_len]
            loss, acc, predict, _ = sess.run(
                [model.loss, model.accuracy, model.predictions, model.train_op],
                feed_dict={model.input_x: input_x, model.input_y: input_y,
                           model.dropout_keep_prob: dropout_keep_prob})
            print("loss:", loss, "acc:", acc, "label:", input_y, "prediction:", predict)
    print('hello world')    
