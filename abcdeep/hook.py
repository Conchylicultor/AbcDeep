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
import functools
import collections
import tensorflow as tf

import abcdeep
import abcdeep.hookcontroller as hookcontroller
from abcdeep.argsutils import ArgParser, ArgGroup
from abcdeep.constant import GraphMode, GraphKey


__all__ = [
    'HookSharedState',
    'InterruptHook',
    'SaverHook',
    'GlobStepCounterHook',
]

# TODO: Refactor into multiple files inside abcdeep/hook/


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
        """
        When this function is called, the arguments have not been parsed yet so
        the constructor code should be moved inside _init instead
        The function can be used eventually to add new arguments function for
        the arg_parse
        """
        super().__init__()
        self.state = None  # Each hook share a common state object

    def _init(self, state, controllers=None):
        """ Contructor of the hook
        This function is called after the arguments have been parsed
        Args:
            state (obj): the shared state of all the hooks
            controller (HookController or List[HookController]): object
                controling when the hook is run (ex: only for test,...). If
                multiples controllers are given, they are apply in the order
                they are added
        """
        self.state = state

        for c in abcdeep.iterify(controllers):
            c.apply(self)


class ModeSelectorHook(AbcHook):
    """ Select the run mode among predetermined ones
    """
    def __init__(self, modes=None, policy=None):
        """
        Args:
            modes (List[str]): the list of modes. If None, use the default
                modes (train, val and test)
            policy (obj): The mode contoller (ex: run 1 validation iteration
                every 10 training iterations and run complete testing set every
                1000 iterations)
        """
        super().__init__()
        self.modes = modes or GraphMode._attr_values
        self.p_choices = {}
        self.policy = policy or hookcontroller.AlternatePolicy()  # TODO: hookcontroller.DefaultPolicy()

    def _init(self, state):
        super()._init(state)
        self.state.curr_mode = self.modes[0]

        # Build the modes
        with tf.name_scope('mode_choice'):
            for m in self.modes:
                self.p_choices[m] = tf.placeholder_with_default(
                    False,
                    shape=(),
                    name='mode_{}'.format(m)
                )

        t_is_train = tf.identity(self.p_choices[GraphMode.TRAIN], name='is_train')
        GraphKey.add_key(GraphKey.IS_TRAIN, t_is_train)

    def before_run(self, run_context):
        """
        """
        self.state.curr_mode = self.policy.run(self.state)

        # TODO: Control when test mode
        return tf.train.SessionRunArgs(None, feed_dict={
            self.p_choices[self.state.curr_mode]: True
        })


class InterruptHook(AbcHook):
    """ Stop the session when a SIGINT is captured
    """
    def _init(self, state):
        super()._init(state)
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
    Will substitue itself to the default scafold by adding itself all saver and
    init ops into the collections tf.GraphKeys.SAVERS and cie.
    Should only be created once per graph.
    """
    # TODO: Also restore params (instead of program)
    # TODO: Also save extra information not contained in the graph (data ?,
    # parser!, program version, glob_step!!...). Use fct ?

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.TRAINING)
    def training_args(parser):
        """ Register the program arguments
        """
        parser.add_argument('--save_every', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        parser.add_argument('--keep_every', type=float, default=0.3, help='if this option is set, a saved model will be keep every x hours (can be a fraction) (Warning: make sure you have enough free disk space or increase save_every)')

    def _init(self, state):
        super()._init(
            state,
            controllers=[
                hookcontroller.OnlyModeController(),
                hookcontroller.EveryXIterController(
                    state.args.save_every,
                    at_first=False,
                ),
            ]
        )
        self.sess = None

        self.saver = None
        self.init_op = None
        self.ready_for_local_init_op = None  # Once the global variables are initialized, initialized local ones
        self.local_init_op = None
        self.ready_op = None

        self.MODEL_EXT = '.index'

    def _save(self):
        print('Saving current model...')
        if not os.path.exists(self.state.model_dir):
            os.makedirs(self.state.model_dir, exist_ok=True)
        self.state.program._save_params()
        model_prefix = self._get_model_prefix()
        self.saver.save(self.sess, model_prefix)  # Put a limit size (ex: 3GB for the model_dir) ?
        print('Model saved: {}'.format(model_prefix))

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
        with tf.name_scope('saver_hook'):
            self.saver = tf.train.Saver(  # TODO: Add an option to delimit a max size ?
                max_to_keep=10,
                keep_checkpoint_every_n_hours=self.state.args.keep_every,
                # pad_step_number=True,  # Pad with 0 the global step
            )
            self.init_op = tf.global_variables_initializer()
            self.ready_for_local_init_op = tf.report_uninitialized_variables(
                tf.global_variables(),
                name='ready_for_local_init_op',
            )
            with tf.name_scope('local_init_op'):
                self.local_init_op = tf.group(
                    tf.local_variables_initializer(),
                    tf.tables_initializer(),  # Why the orignal Scater code initialize tables_initializer here ?
                )
            self.ready_op = tf.report_uninitialized_variables()

        # Necessary to overwrite the default tf.train.Scafold saver:
        tf.add_to_collection(tf.GraphKeys.SAVERS, self.saver)
        tf.add_to_collection(tf.GraphKeys.INIT_OP, self.init_op)
        tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, self.ready_for_local_init_op)
        tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, self.local_init_op)
        tf.add_to_collection(tf.GraphKeys.READY_OP, self.ready_op)
        # TODO: Is there more differences with original saver ??
        #  * Diff with Saver ?
        #  * local_init_op: Why tables_initializer ?
        #  * ready_op: Why report_uninitialized_variables(),
        #  report_uninitialized_resources()  why both variable/ressources ? What
        #  are tensorflow.python.ops.resources ?
        # TODO: Explore the content of GraphKeys.RESOURCES and GraphKeys.LOCAL_RESOURCES

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
            def is_dir_empty(directory):
                """ Return True if the directory contains files
                """
                if not os.path.isdir(directory):
                    return True
                for _, _, files in os.walk(directory):
                    if files:
                        return False
                return True

            if not is_dir_empty(self.state.model_dir):
                raise ValueError('Model not found, but some files are presents. Use --reset to clean the directory')

            print('No model found. Initialising the model...')
            #sess.run(tf.global_variables_initializer()) #.eval()

    def after_run(self, run_context, run_values):
        """ Eventually save the model (every X iterations)
        """
        self._save()

    def end(self, sess):
        """ If training mode, perform an ultimate saving
        """
        self._save()


class GraphSummaryHook(AbcHook):
    """ Save the graph architecture for TensorBoard
    Is only executed once at the first iteration.
    """

    def _init(self, state):
        super()._init(state)
        self.summary = None
        self.save_dir = ''

        self.SUMMARY_DIR = 'graph'
        self.GRAPHDEF_FILENAME = 'graph.pbtxt'

    def begin(self):
        """
        """
        self.save_dir = os.path.join(self.state.model_dir, self.SUMMARY_DIR)
        self.summary = tf.summary.FileWriter(self.save_dir)

    def after_create_session(self, sess, coord):
        """ Restore the model or perform a global initialization
        """
        if self.state.glob_step == 0:
            # Check if files are already presents ?
            tf.train.write_graph(
                sess.graph_def,
                self.save_dir,
                self.GRAPHDEF_FILENAME,
                as_text=True,
            )
            self.summary.add_graph(sess.graph)
            # Also use .add_meta_graph ?? What difference ???


class GlobStepCounterHook(AbcHook):
    """ Increment the global step
    Print a progression bar for each epoch
    """
    def __init__(self):
        """
        """
        super().__init__()

        self.bar_gen = None  # Object manager which allows to restore the standard terminal output after using tqdm
        self.bar = None
        self.epoch_size = 0

    def _init(self, state):
        super()._init(
            state,
            controllers=hookcontroller.OnlyModeController()
        )

    def after_create_session(self, sess, coord):
        """
        """
        self.bar_gen = abcdeep.tqdm_redirector()
        self.bar = next(self.bar_gen)

    def before_run(self, run_context):
        """
        """
        # TODO: Communicate with data_loader to get the epoch size
        # TODO: Close/recreate bar when finishing an epoch

    # TODO: Only called during training mode
    def after_run(self, run_context, run_values):
        self.state.glob_step += 1
        self.bar.update(1)

    def end(self, sess):
        """
        """
        try:  # TODO: Bad code: replace tqdm_redirector() by better manager
            next(self.bar_gen)
        except StopIteration:
            pass


class PrintLossHook(AbcHook):
    """
    """

    def _init(self, state):
        """
        """
        # TODO: Avoid hardcoded step
        super()._init(state, controllers=[
            hookcontroller.OnlyModeController(),
            hookcontroller.EveryXIterController(50)]
        )

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches={GraphKey.LOSS: GraphKey.get_key(GraphKey.LOSS)}
        )

    def after_run(self, run_context, run_values):
        print('Loss at {iter}: {curr:.4f} (avg={avg:.4f})'.format(
            iter=self.state.glob_step,
            curr=run_values.results[GraphKey.LOSS],
            avg=0.0,
        ))


class HyperParamSchedulerHook(AbcHook):
    """ Control a hyperparameter schedule
    """
    def __init__(self, name, default):
        """
        """
        # TODO: Allow special variable for test mode (ex: set dropout at 1.0)
        super().__init__()

        self.name = name
        self.p_param = None
        self.default = default

        # TODO: Improve function (more parameters, more choice, default
        # scheduler, different param for test,...)!

        @ArgParser.regiser_args(ArgGroup.TRAINING)
        def training_args(parser, name, default):
            """ Register the program arguments
            """
            parser.add_argument(
                '--{}'.format(name),
                type=float,
                default=default,
                help='control the {} parameter'.format(name),
            )
        training_args = functools.wraps(training_args)(
            functools.partial(
                training_args,
                name=name,
                default=default,
            )
        )

        setattr(  # TODO: The class will be parsed twice (modify arg_parse to flag the methods aldready parsed ?)
            HyperParamSchedulerHook,
            '{}_args'.format(name),
            staticmethod(training_args),
        )

    def _init(self, state):
        """
        """
        super()._init(state)
        self.p_param = tf.placeholder(tf.float32, shape=(), name=self.name)
        GraphKey.add_key(self.name, self.p_param)  # TODO: Better collision name handing (create custom methods ??)

    def before_run(self, run_context):
        """
        """
        return tf.train.SessionRunArgs(None, feed_dict={self.p_param: self.default})
