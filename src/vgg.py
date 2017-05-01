#!/usr/bin/env python
# encoding: utf-8

"""Implement main model here."""

# import tensorflow as tf


class vgg(object):
    """Main model."""

    def __init__(self, config):
        """initializer."""
        self.batch_size = config.batch_size
        self.params_dir = config.params_dir
