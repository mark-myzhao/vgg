"""Evaluate the performance of model with AP."""
import vgg
import config
import read_data

import tensorflow as tf


def calculate_ap():
    """Calculate AverP"""
    pass


def main():
    """Main Operation."""
    t_config = config.Config()
    with tf.Graph().as_default():
        reader = read_data.ImageReader('../data/JPEGImages/',
                                       '../data/labels/', t_config)
        # init model
        model = vgg.Vgg(t_config)

        # feed feedforward
        predict = model.build_model(True)

        # initializing operation
        init_op = tf.global_variables_initializer()

        # create saver
        saver = tf.train.Saver(max_to_keep=100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            # restore model
            sess.run(init_op)
            model.restore(sess, saver, t_config.load_filename)

            for idx in xrange(5):
                with tf.device("/cpu:0"):
                    imgs, labels, name_list = reader.get_batch(False, idx)

                # feed data into the model
                feed_dict = {
                    model.images: imgs
                    }
                with tf.device(t_config.gpu):
                    # run the training operation
                    res = sess.run(predict, feed_dict=feed_dict)
                    print(res)


if __name__ == '__main__':
    main()
