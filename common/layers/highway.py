#coding:utf-8


def highway(inputs, num_outputs, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    '''
    highway network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate
    '''
    size = int(num_outputs)
    with tf.variable_scope(scope):
        for index in range(num_layers):
            g = f(tf.contrib.layers.linear(inputs, size, scope='highway_lin_%d' % index))

            t = tf.sigmoid(tf.contrib.layers.linear(inputs, size, scope='highway_gate_%d' % index) + bias)

            outputs = t * g + (1 - t) * inputs
            inputs = outputs

    return outputs
