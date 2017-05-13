# Copyright 2017 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Centralize the summaries computations

This file's goal is to add some features to the original tensorflow summary:
 * Centralize switch to easily disable a particular type of summary (without
 having to modify the model code)
 * Allows prioritisation of some summary (we don't want to compute the
gradient histograms at each iteration)
 * Plot training/testing loss curve on the same tensorboard graph (needs two
 FileWriter)
 * Wrap all summary inside a name_scope (for tensorboard) ?
 * Particular schedule for images summary: At the begining, record images often
 (to see if the networt is training correctly). After don't need to record as
 many
 * Keep event files separate between summary types (otherwise gigantic files
 70+GB for summary with images)
"""

import os
import tensorflow as tf

import abcdeep.hook as hook
from abcdeep.constant import GraphMode


class SummaryKeys:
    """ Wrapper around the tf summary module
    Add the summary ops into collections
    """
    HIGH = 'summary_high'
    LOW = 'summary_low'
    # TODO: Some convenience method to easily capture the tf.layers summaries
    # ops
    # TODO: Different summaries for the different queues (two different
    # summary operators val and train for the images)
    # TODO: Wrap summary under a unique 'summary' name scope ?

    def add_scalar(name, tensor, priority=None):
        if priority is None:
            priority = SummaryKeys.HIGH
        tf.summary.scalar(name, tensor, collections=[priority])


class SummaryHook(hook.AbcHook):
    """ Record the training/validation curve at each step
    """
    # TODO: Record train/test into separate folders
    # TODO: Record images in separate folders (allows to remove them without
    # erasing the training curves)
    # TODO: Add args parser to control the iterations for which the summary are
    # run (compute_summary_train_every) ? Better to use default values ?

    def _init(self, state):
        super()._init(state)
        self.summary_train = None
        self.summary_val = None
        self.summary_test = None

        self.op_high = None

        self.SUMMARY_TRAIN_DIR = 'train'  # Warning: Collision with other hooks ?
        self.SUMMARY_VAL_DIR = 'val'
        self.SUMMARY_TEST_DIR = 'test'
        self.SUMMARY_OP = 'summary_op'

    def begin(self):
        """
        """
        def gen_save_dir(suffix):
            return os.path.join(self.state.model_dir, suffix)

        self.summary_train = tf.summary.FileWriter(gen_save_dir(self.SUMMARY_TRAIN_DIR))
        self.summary_val = tf.summary.FileWriter(gen_save_dir(self.SUMMARY_VAL_DIR))
        self.summary_test = tf.summary.FileWriter(gen_save_dir(self.SUMMARY_TEST_DIR))

        # TODO: Capture all collections/priority ops
        self.op_high = tf.summary.merge_all(key=SummaryKeys.HIGH)

    def before_run(self, run_context):
        """
        """
        # TODO: Control the priorities (run only every x glob_steps)
        return tf.train.SessionRunArgs({self.SUMMARY_OP: self.op_high})

    def after_run(self, run_context, run_values):
        """
        """
        if self.state.curr_mode == GraphMode.TRAIN:
            curr_summary = self.summary_train
        elif self.state.curr_mode == GraphMode.VAL:
            curr_summary = self.summary_val
        # TODO: How to handle different tests modes ?
        #elif self.state.curr_mode == GraphMode.TEST:
        #    curr_summary = self.summary_test

        # TODO: Control when test mode
        curr_summary.add_summary(
            run_values.results[self.SUMMARY_OP],
            self.state.glob_step
        )
