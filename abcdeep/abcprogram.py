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

""" The main program with the taining loop
"""

import os
import platform  # Print Python version
import configparser
import tensorflow as tf

import abcdeep
import abcdeep.hook as hook
from abcdeep.constant import GraphKey
from abcdeep.summary import SummaryHook
from abcdeep.argsutils import ArgParser, ArgGroup
from abcdeep.otherutils import cprint, TermMsg


__all__ = ['ProgramInfo', 'AbcProgram']


class ProgramInfo:
    """ Define constants associated with the program
    """
    def __init__(self, name, version):
        """
        Args:
            name (str): the program name
            version (str): the program version. Is not used for compatibility checking
        """
        self.name = name
        self.version = version


def print_title(text, size=64):
    """ Print the given text with a better ascii formatting
    """
    print('#' * size)
    print('#' + ' ' * (size - 2) + '#')
    print('# {text:^{pad}} #'.format(text=text, pad=(size - 4)))
    print('#' + ' ' * (size - 2) + '#')
    print('#' * size)


class DebugMode(abcdeep.OrderedAttr):
    INFO = 'info'
    DEBUG = 'debug'
    WARN = 'warn'
    ERROR = 'error'
    FATAL = 'fatal'


class AbcProgram:

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.GLOBAL)
    def global_args(parser):
        """ Register the program arguments
        """
        parser.add_argument('--models_dir', type=str, default=None, help='root folder in which the models are saved and loaded')
        parser.add_argument('--model_tag', type=str, default=None, help='tag to differentiate which model to store/load')
        parser.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        parser.add_argument('--debug_mode', **DebugMode.arg_choices(), help='verbosity of the tensorflow debug mode')

    #@abcdeep.abstract
    def _customize_args(self, arg_parser):
        """ Allows child classes to customize the argument parser
        Can overwrite default arguments and add new custom ones. Is called after
        having registered all standard arguments.
        Args:
            arg_parser (ArgParser): The program ArgParser object
        """
        pass

    #@abcdeep.abstract
    def _add_modes(self):
        """ This function is called to add the modes (training, testing,...).
        Is called after the argument parsing and model creation.
        """
        # TODO: Pb: need the argument to build the hooks but the hooks need to
        # register their arguments. Chicken-Egg problem. When adding an
        # hyperparametter (learning rate, dropout,...) with a schedule using a
        # hook, the same hook can be used multiple times but with different args.
        # Sol: The modes and hooks are constructed initially before parse_args but
        # can modify themselves aftewards (ex: when create_session is called)
        pass

    #@abcdeep.abstract
    def _get_all_hooks(self):
        """ Define the hooks added to the session
        Will call _get_hooks_before and _get_hooks_after to customize the added
        hooks
        This function can be overwritten to modify the default hooks.
        Return:
            List[Hook]: The list of all hooks
        """
        # WARNING: Calling order is important (TODO: Doc for each hook which define what should be run after/before)
        return [
            hook.ModeSelectorHook(),  # First hyperparameters
            hook.HyperParamSchedulerHook(GraphKey.LEARNING_RATE, 0.001),
            # hook.HyperParamSchedulerHook(GraphKey.DROPOUT, 0.8),
            *self._get_hooks_before(),  # Custom hooks TODO: Use variable instead of function instead ?
            self.dataconnector_cls(),  # Then input connector
            self.model_cls(),  # Finally the main model
            *self._get_hooks_after(),  # Custom hooks TODO: Remove ? Is _get_hooks_before enough ?
            hook.PrintLossHook(),
            hook.SaverHook(),  # Saver has to be last one to modify graph (for the saving/init ops)
            SummaryHook(),  # Summary is added after every summary have been added
            hook.GraphSummaryHook(),  # Summary is after the saver (because of reset)
            hook.GlobStepCounterHook(),  # glob step is called at the end
            hook.InterruptHook(),
        ]

    #@abcdeep.abstract
    def _get_hooks_before(self):
        """ Allows to add some hooks to the default ones
        The hooks are added before the model is create so can be used to add
        HyperParamSchedulerHook for instance
        Return:
            List[Hook]: A the hooks to add to the default ones
        """
        return []

    #@abcdeep.abstract
    def _get_hooks_after(self):
        """ Allows to add some hooks to the default ones
        The hooks are added after the data and model
        Return:
            List[Hook]: A the hooks to add to the default ones
        """
        return []

    def __init__(self, program_info, model=None, dataconnector=None):
        """
        Args:
            program_info (ProgramInfo): contains the name and version of the program
            model (class): Class of the model to build
            dataconnector (class): Intermediate class between the data and the
                model. Used for loading/saving the input/output
        """
        self.program_info = program_info
        self.args = None
        self.arg_parser = None

        self.model_cls = model
        self.model_dir = ''
        self.dataconnector_cls = dataconnector

        self.hooks = []
        self.hook_state = None

        self.MODELS_DIR = 'save'
        self.MODEL_DIR_PREFIX = 'model'
        self.CONFIG_FILENAME = 'config.ini'


    def add_mode(self):
        pass

    def run(self, args=None):
        """ Launch the program with the given arguments
        Args:
            args (List[str]): If None, will take the command line argument
        """
        # Global infos on the program
        print_title('Welcome to {} v{} !'.format(
            self.program_info.name,
            self.program_info.version
        ))
        print()
        print('Python v{} - TensorFlow v{} ({})'.format(
            platform.python_version(),
            tf.__version__,
            platform.platform(),
        ))
        print()

        self.hooks = self._get_all_hooks()

        # Parse the command lines arguments
        self.arg_parser = ArgParser()
        self.arg_parser.register_cls(type(self))
        #self.arg_parser.register_cls(self.model_cls)
        #self.arg_parser.register_cls(self.dataconnector_cls)
        for h in self.hooks:  # Register all args from all added hooks
            # TODO: What if the same hook is added multiple times. How to handle
            # args name collisions ?
            self.arg_parser.register_cls(type(h))  # TODO: Register instance instead of type
        self._customize_args(self.arg_parser)
        self.args = self.arg_parser.parse_args(args)

        # Compute some global variables and save/restore the previous parameters
        self._set_dirs()
        self.hook_state = hook.HookSharedState(self)  # Called before _restore but after _set_dirs
        if self.args.reset:
            self._reset()
        self._restore_args()
        self._set_tf_verbosity()  # Set general debug mode

        self.arg_parser.print_args()

        # Launch the associated mode
        self._main()

    def _main(self):
        """
        """
        # TODO: Set mode/initial

        # TODO: Single Hook which encapsulate and control all hooks (run for good mode,
        # for good iteration, forward parameters (glob_step,...)) ?

        cprint('############### Initialization ###############', color=TermMsg.H1)

        print('Building graph...')
        for hook in self.hooks:
            h = hook._init(self.hook_state)

        print('Launching session...')
        # The queue_runners are integrated at MonitoredSession
        with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(),  # Scaffold.finalize will look at the collections to use the operators added by the hooks
            ),
            hooks=self.hooks
        ) as sess:
            # TODO: Correct additional line bug when printing whith both color and tqdm
            cprint('############### Session launched ###############', color=TermMsg.H1)

            while not sess.should_stop():
                # TODO: Format output with tqdm (can probably be done using hook ?)
                # TODO: Set current mode for the loop
                sess.run([])  # Empty loop (the fetches are added by the hooks)

        print('The End! Thank you for using this program.')

    def _set_dirs(self):
        """ Compute the current paths for the model and data
        By default, use location relative to the root directory:
          * For model: save/model-<mode_tag>
          * For data:
        """
        # Compute the current model path (Use the current working directory if
        # None set)
        default_model_dir = os.path.join(os.getcwd(), self.MODELS_DIR)
        self.model_dir = self.args.models_dir or default_model_dir
        self.model_dir = os.path.join(self.model_dir, self.MODEL_DIR_PREFIX)
        if self.args.model_tag:
            self.model_dir += '-' + self.args.model_tag

        # Compute the data paths

    def _reset(self):
        """ Delete all the model dir content.

        Warning: No confirmation is asked. All subfolders will be deleted
        """
        assert self.args.reset
        if os.path.exists(self.model_dir):
            cprint(
                'Warning: Delete all content of {}'.format(self.model_dir),
                color=TermMsg.WARNING
            )
            for root, dirs, files in os.walk(self.model_dir, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    print('Removing {}'.format(file_path))
                    os.remove(file_path)
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

    def _restore_args(self):
        """ Load the some values associated with the current model, like the
        current glob_step value.
        Is one of the first function called because it initialize some
        variables used on the rest of the program.

        Warning: If you modify this function, make sure to mirror the changes
        in _save_args
        """
        # If there is a previous model, restore some parameters
        config_name = os.path.join(self.model_dir, self.CONFIG_FILENAME)
        if os.path.exists(config_name):
            print('Warning: Restoring previous configuration from {}'.format(config_name))
            config = configparser.ConfigParser()
            config.read(config_name)

            # TODO: Check the program name/version ??
            # TODO: Restore glob_step (and other hooks parameters)
            self.hook_state.restore_args(config)
            self.arg_parser.restore_args(config)

    def _save_params(self):
        """ Save the params of the model, like the current glob_step value
        Warning: if you modify this function, make sure the changes mirror _restore_args
        """
        config = configparser.ConfigParser()
        # TODO: Save the program name/version ??
        self.hook_state.save_args(config)
        self.arg_parser.save_args(config)

        config_name = os.path.join(self.model_dir, self.CONFIG_FILENAME)
        with open(config_name, 'w') as config_file:
            config.write(config_file)

    def _set_tf_verbosity(self):
        """ Set the debug mode for tensorflow
        """
        tf.logging.set_verbosity(
            getattr(tf.logging, self.args.debug_mode.upper())
        )
