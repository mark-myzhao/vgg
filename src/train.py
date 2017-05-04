"""Train model."""
import vgg
import config
import read_data

import os
from datetime import datetime
import tensorflow as tf


def main():
    """Main operations."""
    t_config = config.Config()
    with tf.Graph().as_default():
        reader = read_data.ImageReader('./data/JPEGImages/',
                                       './data/labels/', t_config)

        # init model
        model = vgg.Vgg(t_config)

        # feed feedforward
        model.build_model(True)

        # return loss
        loss = model.loss()

        # training operation
        train_op = model.train_op(loss, model.global_step)

        # initializing operation
        init_op = tf.global_variables_initializer()

        # create saver
        saver = tf.train.Saver(max_to_keep=100)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            # initialize parameters or restore from previous model
            if not os.path.exists(t_config.params_dir):
                os.makedirs(t_config.params_dir)
            if os.listdir(t_config.params_dir) == [] or t_config.initialize:
                print("Initializing Network")
                sess.run(init_op)
            else:
                sess.run(init_op)
                model.restore(sess, saver, t_config.load_filename)

            merged = tf.summary.merge_all()
            logdir = os.path.join(t_config.logdir,
                                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            writer = tf.summary.FileWriter(logdir, sess.graph)

            # start training
            for idx in xrange(t_config.max_iteration):
                with tf.device("/cpu:0"):
                    imgs, labels, name_list = reader.get_batch()

                # feed data into the model
                feed_dict = {
                    model.images: imgs,
                    model.labels: labels
                    }
                with tf.device(t_config.gpu):
                    # run the training operation
                    sess.run(train_op, feed_dict=feed_dict)

                with tf.device('/cpu:0'):
                    # write summary
                    if (idx + 1) % t_config.summary_iters == 0:
                        tmp_global_step = model.global_step.eval()
                        summary = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(summary, tmp_global_step)
                    # save checkpoint
                    if (idx + 1) % t_config.checkpoint_iters == 0:
                        tmp_global_step = model.global_step.eval()
                        model.save(sess, saver, t_config.save_filename,
                                   tmp_global_step)


if __name__ == '__main__':
    main()
