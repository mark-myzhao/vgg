#!/usr/bin/env python
# encoding: utf-8

"""Implement main model here."""

import tensorflow as tf
# import numpy as np


class vgg(object):
    """Main model."""

    def __init__(self, config):
        """Initialize with config object."""
        # TODO
        self.batch_size = config.batch_size
        self.params_dir = config.params_dir

    def train_op(self, total_loss, global_step):
        """Get train op."""
        # return train_op
        # TODO
        pass

    def loss(self):
        """Get loss op."""
        # TODO
        pass

    def save(self, sess, saver, filename, global_step):
        """Save to check point."""
        # TODO
        pass

    def restore(self, sess, saver, filename):
        """Restore from a check point."""
        # TODO
        pass

    def build_model(self, is_train):
        """Get builded vgg model."""
        fc_is_train = is_train & True
        with tf.name_scope("original_images"):
            self._image_summary(self.images, 1)
        out_fc = self.cnn_fc(self.images, fc_is_train, 'fc')
        self.add_to_euclidean_loss(self.batch_size, out_fc, self.coords, 'fcn')
        return out_fc

    def cnn_fc(self, input_, is_train, name):
        """Build vgg model D with 16 layers."""
        with tf.variable_scope(name):
            # TODO Build Model Here
            pass
        # return final_fc

    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        """Add to euclidean loss."""
        flatten_labels = tf.reshape(labels, [batch_size, -1])
        flatten_predicts = tf.reshape(predicts, [batch_size, -1])

        with tf.name_scope(name):
            euclidean_loss = tf.sqrt(tf.reduce_sum(
                tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
            euclidean_loss_mean = tf.reduce_mean(euclidean_loss,
                                                 name='euclidean_loss_mean')

        tf.add_to_collection("losses", euclidean_loss_mean)

    def fc_layer(self, bottom, out_num, name, is_BN, trainable):
        """Construct fully-connected layer."""
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            weights = self._variable_with_weight_decay(
                "weights",
                shape=[flatten_bottom.get_shape()[-1], out_num],
                stddev=self.stddev,
                wd=self.wd,
                trainable=trainable
            )
            mul = tf.matmul(flatten_bottom, weights)
            biases = self._variable_on_cpu(
                'biases', [out_num],
                tf.constant_initializer(0.0),
                trainable
            )
            pre_activation = tf.nn.bias_add(mul, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(top)
        return top

    def conv_layer(self, bottom, kernel_size, out_channel, name, is_BN,
                   trainable):
        """Construct conv-layer."""
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay(
                    "weights",
                    shape=[kernel_size, kernel_size, bottom.get_shape()[-1],
                           out_channel],
                    stddev=self.stddev,
                    wd=self.wd,
                    trainable=trainable
            )
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding="SAME")
            biases = self._variable_on_cpu(
                'biases', [out_channel],
                tf.constant_initializer(0.0),
                trainable
            )
            pre_activation = tf.nn.bias_add(conv, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(top)
        return top

    def _variable_on_cpu(self, name, shape, initializer, trainable):
        """Get a variable on cpu."""
        with tf.device('/cpu:0'):
            dtype = tf.float16 if self.use_fp16 else tf.float32
            var = tf.get_variable(name, shape, initializer=initializer,
                                  dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable):
        dtype = tf.float16 if self.use_fp16 else tf.float32
        var = self._variable_on_cpu(
            name, shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
            trainable
        )
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name='weights_loss')
            tf.add_to_collection("losses", weight_decay)
        return var

    # Summary
    def _activation_summary(self, x):
        name = x.op.name
        tf.summary.histogram(name + '/activations', x)
        tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(x))

    def _image_summary(self, x, channels):
        def sub(batch, idx):
            name = x.op.name
            tmp = x[batch, :, :, idx] * 255
            tmp = tf.expand_dims(tmp, axis=2)
            tmp = tf.expand_dims(tmp, axis=0)
            tf.summary.image(name + '-' + str(idx), tmp, max_outputs=100)
        if (self.batch_size > 1):
            for idx in xrange(channels):
                # the first batch
                sub(0, idx)
                # the last batch
                sub(-1, idx)
        else:
            for idx in xrange(channels):
                sub(0, idx)

    def _loss_summary(self, loss):
        tf.summary.scalar(loss.op.name + " (raw)", loss)

    def _fm_summary(self, predicts):
        with tf.name_scope("fcn_summary"):
            self._image_summary(self.labels, self.points_num)
            tmp_predicts = tf.nn.relu(predicts)
            self._image_summary(tmp_predicts, self.points_num)
