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

import tensorflow as tf

import abcdeep
import abcdeep.hook as hook
from abcdeep.argsutils import ArgParser, ArgGroup


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


class DebugMode(abcdeep.OrderedAttr):
    INFO = 'info'
    DEBUG = 'debug'
    WARN = 'warn'
    ERROR = 'error'
    FATAL = 'fatal'


class AbortLoop(Exception):
    """ Raised by the hooks to skip the current iteration
    """


class AbortProgram(Exception):
    """ Raised by the hooks to end the program
    """

class AbcProgram:

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.GLOBAL)
    def global_args(parser):
        """ Register the program arguments
        """
        parser.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')
        parser.add_argument('--debug_mode', **DebugMode.arg_choices(), help='verbosity of the tensorflow debug mode')

    #@abcdeep.abstract
    def _customize_args(self, arg_parser):
        """ Allows child classes to customize the argument parser
        Can overrite default arguments and add new custom ones. Is called after
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
        pass

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

        self.model_cls = model
        self.dataconnector_cls = dataconnector

    def add_mode(self):
        pass

    def run(self, args=None):
        """
        """
        # Global infos on the program
        print('Welcome to {} v{} !'.format(
            self.program_info.name,
            self.program_info.version
        ))
        print()
        print('TensorFlow detected: v{}'.format(tf.__version__))
        print()

        # Parse the command lines arguments
        arg_parser = ArgParser()
        arg_parser.register_cls(type(self))
        arg_parser.register_cls(self.model_cls)
        arg_parser.register_cls(self.dataconnector_cls)
        # TODO: Register all args from all added hooks
        self._customize_args(arg_parser)
        self.args = arg_parser.parse_args(args)
        arg_parser.print_args()

        # Set general debug mode
        self.set_tf_verbosity()

        # Launch the associated mode
        self.main()

    def main(self):
        """
        """
        # TODO: Set mode/initial

        # TODO: Single Hook which encapsulate and control all hooks (run for good mode,
        # for good iteration, forward parameters (glob_step,...)) ?

        interrupt_hook = hook.InterruptHook()

        with tf.train.MonitoredSession(
            session_creator=tf.train.ChiefSessionCreator(),
            hooks=[interrupt_hook],  #hooks=[saver_hook, summary_hook]
        ) as sess:

            with abcdeep.interrupt_handler() as h:  # Capture Ctrl+C
                interrupt_hook.interrupt_state = h

                while not sess.should_stop():
                    # TODO: Format output with tqdm (can probably be done using hook ?)
                    # TODO: Set current mode for the loop
                    sess.run([])  # Empty loop (the fetches are added by the hooks)
                    #except AbortLoop:  # Abort the current loop (ex: invalid input)
                    #    pass
                    #except AbortProgram:  # Abort program
                    #    sess.request_stop()

        print('The End! Thank you for using our program.')

    def set_tf_verbosity(self):
        """ Set the debug mode for tensorflow
        """
        tf.logging.set_verbosity(
            getattr(tf.logging, self.args.debug_mode.upper())
        )
