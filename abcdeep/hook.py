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

""" Define the session run hooks main structure
"""

import tensorflow as tf


__all__ = ['InterruptHook']


class HookSharedState:
    """ Container for the variables shared amongs the hooks
    """
    def __init__(self):
        self.args = None  # Program parameters
        self.model_dir = ''  # Current model directory (for saving/restoring)
        self.curr_mode = None  # Mode for the current iteration
        self.glob_step = 0  # Number of training iterations
        self.interrupt_state = None  # From a interrupt_handler context manager


class InterruptHook(tf.train.SessionRunHook):
    """ Stop the session when a SIGINT is captured
    """
    def __init__(self):
        super().__init__()
        # Singal handler state (InterruptState) from a interrupt_handler
        # context manager
        self.interrupt_state = None

    def after_run(self, run_context, run_values):
        if self.interrupt_state.interrupted:
            print('Stop requested, exiting...')
            run_context.request_stop()


class SaverHook(tf.train.SessionRunHook):
    """ Manage the model
    """
    # TODO: Also save extra information not contained in the graph (data ?,
    # parser!, program version, glob_step!!...). Use fct ?
    def __init__(self):
        super().__init__()
        self.glob_step = 0  # TODO: as interrupt_state, shared for all hooks
        self.save_every = 200  # TODO: args instead
        self.parser = None  #

    def save(self):
        pass

    def restore(self):
        pass

    def after_create_session(session, coord):
        """ Restore the model or perform a global initialization
        """

    def after_run(self, run_context, run_values):
        """ Eventually save the model (every X iterations)
        """

    def end(session):
        """ If training mode, perform an ultimate saving
        """


class GlobStepCounterHook(tf.train.SessionRunHook):
    """ Increment the global step
    """
    # TODO: Only called during training mode !
    def __init__(self):
        super().__init__()


    def after_run(self, run_context, run_values):
        self.state.glob_step += 1



class AbcHook(tf.train.SessionRunHook):
    """ Base class for the hooks
    The hooks are similar to tf.train.SessionRunHook with some minor changes:
      * More methods are available to run special actions at specific times (ex:
      when the program exit)
      * The hooks can chontrol when it is executed (for which mode (train/val),
      every x iterations, every x seconds)
    """

    def __init__(self):
        self.state = None  # Each hook share a common state object
        # TODO: The hook should have access to:
        #  * glob_step
        #  * current GraphMode


class PrintLossHook(AbcHook):
    """
    """
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches={GraphKey.LOSS: GraphKey.get_key(GraphKey.LOSS)}
        )

    def after_run(self, run_context, run_values):
        print('Loss (iter {}): {} ()')
