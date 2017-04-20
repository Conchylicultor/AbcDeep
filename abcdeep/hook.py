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
The current hooks are:
 * SaverHook: Save/Restore the models/params
 * InterruptHook: Exit the program if Ctrl+C is detected
 * GlobStepCounterHook: Increment the counter at each training loop
"""

import os
import tensorflow as tf

import abcdeep
from abcdeep.argsutils import ArgParser, ArgGroup


__all__ = [
    'HookSharedState',
    'InterruptHook',
    'SaverHook',
    'GlobStepCounterHook',
]


class HookSharedState:
    """ Container for the variables shared amongs the hooks
    """
    def __init__(self, program):
        """ Initialze the shared state.
        Warning: Will copy some information (args, path,...) from the program
        class so the function has to be called after those have been initialized
        """
        # Global constants
        self.args = program.args  # Program parameters
        self.model_dir = program.model_dir  # Current model directory (for saving/restoring)
        self.program = program
        # Shared variables
        self.curr_mode = None  # Mode for the current iteration
        self.glob_step = 0  # Number of training iterations

    def restore_args(self, config):
        """ Restore the hook state
        Args:
            config (obj): configparser object
        """
        self.glob_step = config['Hook'].getint('glob_step')
        # TODO: How to avoid conflict with arg_parser names (delete after havig restored ?)?
        # TODO: How to print the restored hooks params ?
        del config['Hook']  # HACK: Avoid collisions with arg_parser

    def save_args(self, config):
        """
        All arguments have to implement __str__. The program just try a
        naive conversion.
        Args:
            config (obj): configparser object
        """
        config['Hook'] = {}
        config['Hook']['glob_step'] = str(self.glob_step)


class AbcHook(tf.train.SessionRunHook):
    """ Base class for the hooks
    The hooks are similar to tf.train.SessionRunHook with some minor changes:
      * The hook have access to a shared state which contains for instance the
      current glob_step
      * The hooks can chontrol when it is executed (for which mode (train/val),
      every x iterations, every x seconds)
    """

    def __init__(self):
        super().__init__()
        self.state = None  # Each hook share a common state object


class InterruptHook(AbcHook):
    """ Stop the session when a SIGINT is captured
    """
    def __init__(self):
        super().__init__()
        self.handler = None
        self.interrupt_state = None

    def after_create_session(self, session, coord):
        """ Start the interrupt_handler context manager
        """
        self.handler = abcdeep.interrupt_handler()
        self.interrupt_state = self.handler.__enter__()

    def after_run(self, run_context, run_values):
        """ At the end of the iteration, we eventually request a stop
        """
        # Singal handler state (InterruptState) from a interrupt_handler
        # context manager
        if self.interrupt_state.interrupted:
            print('Stop requested, exiting...')
            run_context.request_stop()

    def end(self, session):
        """ Exit the interrupt_handler context manager
        """
        self.handler.__exit__(None, None, None)


class SaverHook(AbcHook):
    """ Manage the model
    """
    # TODO: Also restore params (instead of program)
    # TODO: Also save extra information not contained in the graph (data ?,
    # parser!, program version, glob_step!!...). Use fct ?

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.TRAINING)
    def global_args(parser):
        """ Register the program arguments
        """
        parser.add_argument('--save_every', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        parser.add_argument('--keep_every', type=float, default=0.3, help='if this option is set, a saved model will be keep every x hours (can be a fraction) (Warning: make sure you have enough free disk space or increase save_every)')

    def __init__(self):
        super().__init__()
        self.saver = None
        self.sess = None

        self.MODEL_EXT = '.index'

    def _save(self):
        print('Saving current model...')
        if not os.path.exists(self.state.model_dir):
            os.makedirs(self.state.model_dir, exist_ok=True)
        self.state.program._save_params()
        self.saver.save(self.sess, self._get_model_prefix())  # Put a limit size (ex: 3GB for the model_dir) ?
        print('Model saved.')

    def _restore(self):
        pass
        # TODO: Also restore the model

    def _get_model_prefix(self):
        """ Parse the argument to decide were to save/load the model
        This function is called at each checkpoint and the first time the model
        Return:
            str: The path and name were the model need to be saved
        """
        model_name = os.path.join(
            self.state.model_dir,
            'model-{:0>8}'.format(self.state.glob_step),
        )
        return model_name

    def begin(self):
        """
        """
        self.saver = tf.train.Saver(  # TODO: Add an option to delimit a max size ?
            max_to_keep=10,
            keep_checkpoint_every_n_hours=self.state.args.keep_every,
            # pad_step_number=True,  # Pad with 0 the global step
        )

    def after_create_session(self, sess, coord):
        """ Restore the model or perform a global initialization
        """
        self.sess = sess
        # The tensorflow checkpoint paths are saved in absolute path which
        # creates some problem when sharing the models. For this reason, it's
        # better not use tf.train.latest_checkpoint and manually check the model
        # file existance using a HACK
        checkpoint = self._get_model_prefix()

        if os.path.exists(checkpoint + self.MODEL_EXT):  # HACK to check model existance
            print('Restoring model from {}'.format(checkpoint))
            self.saver.restore(
                sess,
                checkpoint,
            )
        else:
            if os.path.isdir(self.state.model_dir) and os.listdir(self.state.model_dir):
                raise ValueError('Model not found, but some files are presents. Use --reset to clean the directory')

            print('No model found. Initialising the model...')
            #sess.run(tf.global_variables_initializer()) #.eval()

    def after_run(self, run_context, run_values):
        """ Eventually save the model (every X iterations)
        """

    def end(self, sess):
        """ If training mode, perform an ultimate saving
        """
        self._save()


class GlobStepCounterHook(AbcHook):
    """ Increment the global step
    """
    # TODO: Only called during training mode
    def after_run(self, run_context, run_values):
        self.state.glob_step += 1


class PrintLossHook(AbcHook):
    """
    """
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches={GraphKey.LOSS: GraphKey.get_key(GraphKey.LOSS)}
        )

    def after_run(self, run_context, run_values):
        print('Loss (iter {}): {} ()')