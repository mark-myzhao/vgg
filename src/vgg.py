#!/usr/bin/env python
# encoding: utf-8

"""Implement main model here."""

import tensorflow as tf
# import numpy as np


class Vgg(object):
    """Main model."""

    def __init__(self, config):
        """Initialize with config object."""
        self.global_step = tf.get_variable("global_step", initializer=0,
                                           dtype=tf.int32, trainable=False)
        # TODO
        self.wd = config.wd
        self.stddev = config.stddev
        self.batch_size = config.batch_size
        self.params_dir = config.params_dir
        self.channel_num = config.channel_num
        self.moving_average_decay = config.moving_average_decay
        self.use_fp16 = config.use_fp16
        self.class_num = config.class_num

        self.images = tf.placeholder(
            dtype=tf.float32,
            shape=(self.batch_size, config.img_height, config.img_width,
                   self.channel_num))
        self.labels = tf.placeholder(
            dtype=tf.float32,
            shape=(self.batch_size, config.class_num))

    def train_op(self, total_loss, global_step):
        """Get train op."""
        self._loss_summary(total_loss)

        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(total_loss)
        apply_gradient_op = optimizer.apply_gradients(grads,
                                                      global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op,
                                      variable_averages_op]):
            train_op = tf.no_op(name="train")

        return train_op

    def loss(self):
        """Get loss op."""
        return tf.add_n(tf.get_collection('losses'), name="total_loss")

    def save(self, sess, saver, filename, global_step):
        """Save to check point."""
        path = saver.save(sess, self.params_dir + filename,
                          global_step=global_step)
        print("Save params at " + path)

    def restore(self, sess, saver, filename):
        """Restore from a check point."""
        print("Restore from previous model: ", self.params_dir+filename)
        saver.restore(sess, self.params_dir + filename)

    def build_model(self, is_train):
        """Get builded vgg model."""
        with tf.name_scope("original_images"):
            self._image_summary(self.images, self.channel_num)
        out_fc = self.cnn_fc(self.images, is_train, 'fc')
        self.add_to_cross_entropy(self.batch_size, out_fc, self.labels, 'fcn')
        return out_fc

    def cnn_fc(self, input_, is_train, name):
        """Build vgg model D with 16 layers."""
        is_BN = True

        with tf.variable_scope(name):
            # TODO Build Model Here
            conv1 = self.conv_layer(input_, 3, 64, 'conv1', is_BN, is_train)
            conv2 = self.conv_layer(conv1, 3, 64, 'conv2', is_BN, is_train)
            pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME", name="pool1")

            conv3 = self.conv_layer(pool1, 3, 128, 'conv3', is_BN, is_train)
            conv4 = self.conv_layer(conv3, 3, 128, 'conv4', is_BN, is_train)
            pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME", name="pool2")

            conv5 = self.conv_layer(pool2, 3, 256, 'conv5', is_BN, is_train)
            conv6 = self.conv_layer(conv5, 3, 256, 'conv6', is_BN, is_train)
            conv7 = self.conv_layer(conv6, 3, 256, 'conv7', is_BN, is_train)
            pool3 = tf.nn.max_pool(conv7, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME", name="pool3")

            conv8 = self.conv_layer(pool3, 3, 512, 'conv8', is_BN, is_train)
            conv9 = self.conv_layer(conv8, 3, 512, 'conv9', is_BN, is_train)
            conv10 = self.conv_layer(conv9, 3, 512, 'conv10', is_BN, is_train)
            pool4 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME", name="pool3")

            conv11 = self.conv_layer(pool4, 3, 512, 'conv11', is_BN, is_train)
            conv12 = self.conv_layer(conv11, 3, 512, 'conv12', is_BN, is_train)
            conv13 = self.conv_layer(conv12, 3, 512, 'conv13', is_BN, is_train)
            pool4 = tf.nn.max_pool(conv13, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding="SAME", name="pool3")

            # change 4096 to 1028
            fc1 = self.fc_layer(pool4, 1028, 'fc1', is_BN, is_train)
            if is_train:
                fc1 = tf.nn.dropout(fc1, 0.5)
            fc2 = self.fc_layer(fc1, 1028, 'fc2', is_BN, is_train)
            if is_train:
                fc2 = tf.nn.dropout(fc2, 0.5)
            final_fc = self.final_fc_layer(fc1, self.class_num,
                                           'final_fc', is_train)
            return final_fc

    def add_to_cross_entropy(self, batch_size, predicts, labels, name):
        """Add cross entropy to loss."""
        flatten_labels = tf.reshape(labels, [batch_size, -1])
        flatten_predicts = tf.reshape(predicts, [batch_size, -1])
        with tf.name_scope(name):
            tmp = tf.nn.softmax_cross_entropy_with_logits(
                labels=flatten_labels,
                logits=flatten_predicts)
            cross_entropy_mean = tf.reduce_mean(tmp, name='cross_entropy_mean')
        tf.add_to_collection("losses", cross_entropy_mean)

    def fc_layer(self, bottom, out_num, name, is_BN, trainable):
        """Construct fully-connected layer."""
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name) as scope:
            # initialize weight
            weights = self._variable_with_weight_decay(
                "weights",
                shape=[flatten_bottom.get_shape()[-1], out_num],
                stddev=self.stddev,
                wd=self.wd,
                trainable=trainable)
            # fully-connected
            mul = tf.matmul(flatten_bottom, weights)
            biases = self._variable_on_cpu(
                'biases',
                [out_num],
                tf.constant_initializer(0.0),
                trainable)
            pre_activation = tf.nn.bias_add(mul, biases)
            if is_BN:
                bn_activation = tf.layers.batch_normalization(pre_activation)
                top = tf.nn.relu(bn_activation, name=scope.name)
            else:
                top = tf.nn.relu(pre_activation, name=scope.name)
            self._activation_summary(top)
        return top

    def final_fc_layer(self, bottom, out_num, name, trainable):
        """Final fully-connected layer."""
        flatten_bottom = tf.reshape(bottom, [self.batch_size, -1])
        with tf.variable_scope(name):
            weights = self._variable_with_weight_decay(
                "weights",
                shape=[flatten_bottom.get_shape()[-1], out_num],
                stddev=self.stddev,
                wd=self.wd,
                trainable=trainable)
            mul = tf.matmul(flatten_bottom, weights)
            biases = self._variable_on_cpu(
                'biases',
                [out_num],
                tf.constant_initializer(0.0), trainable)
            top = tf.nn.bias_add(mul, biases)
            # apply softmax to get probability
            top = tf.nn.softmax(top)
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
                    trainable=trainable)
            # calculate conv
            conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding="SAME")
            biases = self._variable_on_cpu(
                'biases', [out_channel],
                tf.constant_initializer(0.0),
                trainable)
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
        """Used to punish large weights, avoiding overfitting."""
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
        tf.summary.scalar(name + '/min', tf.reduce_min(x))
        tf.summary.scalar(name + '/max', tf.reduce_max(x))
        tf.summary.scalar(name + '/mean', tf.reduce_mean(x))

    def _image_summary(self, x, channels):
        def sub(batch, idx):
            name = x.op.name
            tmp = x[batch, :, :, idx] * 255
            tmp = tf.expand_dims(tmp, axis=2)
            tmp = tf.expand_dims(tmp, axis=0)
            tf.summary.image(name + '-' + str(idx), tmp, max_outputs=100)
        if self.batch_size > 1:
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
