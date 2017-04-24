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

""" Extention of argparse to automatically save/restore command line arguments
The other classes can easily add new arguments to the default ones
"""

import sys
import ast  # For literal eval
import collections
import argparse

import abcdeep.otherutils as otherutils
from abcdeep.otherutils import cprint, TermMsg


# TODO: Create new class allowing to add subprograms (with .add_subparsers). Should
# allow to define a default subparser to use if no subprogram is defined. This
# can be done using the dest argument of add_subparsers and test it after the
# parse_args call (pb is that default help won't be printed)


class ArgGroup(otherutils.OrderedAttr):
    GLOBAL = 'Global'  # Meta arguments: not restored by the program
    DATASET = 'Dataset'  # Network input options
    PREPROCESSING = 'Preprocessing'  # Dataset Creation: done once for all models
    SIMULATOR = 'Simulator'  # RL environement options (recording,...)
    NETWORK = 'Network'  # Network architecture options (layers, hidden params,...)
    TRAINING = 'Training'  # (learning rate, batch size,...)


class ArgParser:
    FCT_FLAG = '_arg_flag'

    def __init__(self):
        self.parser = argparse.ArgumentParser()  # TODO: Allow to use add_subparsers
        self.args = None

        # Correspondance structures
        self._groups = {}  # Contains the argparse group object
        self._group2key = collections.OrderedDict(
            [(k, []) for k in ArgGroup._attr_values]
        )
        self._key2group = {}  # Useless ?
        self._key2action = {}

        self._overwritten = {}  # Argument modified by overwrite()
        self._given = set()  # Argument passed by the command line

    @staticmethod
    def regiser_args(group):
        """ Decorator which flag the function as adding new arguments to the parser
        The decorated function has to have the following signature: Fct(parser)
        Args:
            group (str): The group on which add the arguments will be added. If
            the group is not in ArgumentGroup, it will be created
        """
        def decorator(f):
            setattr(f, ArgParser.FCT_FLAG, group)
            return f
        return decorator

    def register_cls(self, cls):
        """ Parse the given class and add the registered arguments to the parser.
        The class must have flagged some of its methods with `regiser_args`
        Args:
            cls: The class to parse
        """
        def gen_members(cls):  # Return also members in parent classes
            for c in reversed(cls.__mro__):  # TODO: If instance passed instead of class, also parse its non static methods
                for v in vars(c):
                    yield v

        for fct in [getattr(cls, v) for v in gen_members(cls)]:
            if hasattr(fct, ArgParser.FCT_FLAG):
                group_name = getattr(fct, ArgParser.FCT_FLAG)
                # Create the group if not exist
                if group_name not in self._groups:
                    self._groups[group_name] = self.parser.add_argument_group(group_name)
                    self._group2key.setdefault(group_name, [])
                # Add arguments to the group
                fct(self._groups[group_name])

    def overwrite(self, **kwargs):
        """ Allow to replace the defaults values of the arguments.
        Must be called before `parse_args`.
        `parse_args` will raise AttributeError if the argument given here
        don't exist
        """
        self._overwritten.update(kwargs)

    def parse_args(self, argv=None):
        """ This is where the command line is actually parsed
        Args:
            argv (List[str]): Customs arguments to parse
        Return:
            argparse.Namespace
        """
        #self.parser.add_argument_group('Global')

        # When parsing the args, the namespaces keys are also saved
        for group_name, group in self._groups.items():
            for action in group._group_actions:  # HACK: Get the action list
                if action.dest == 'help':  # HACK: filter args added by argparse
                    continue
                self._group2key[group_name].append(action.dest)
                self._key2group[action.dest] = group_name
                self._key2action[action.dest] = action
                # The defaults arguments can be overwritten by .overwrite()
                if action.dest in self._overwritten:
                    action.default = self._overwritten[action.dest]
                    del self._overwritten[action.dest]  # Default overwritten

        if len(self._overwritten):
            raise AttributeError('Unkwnown argparse overwritten key: {}'.format(
                list(self._overwritten.keys())[0])
            )

        # Parse the given args
        self.args = self.parser.parse_args(argv)

        # Save the given arguments (TODO: Is there a better way to extract the args ?)
        if argv is None:
            argv = sys.argv

        def filter_arg(a):  # TODO: Support for other prefix_char
            return a.startswith('--')

        for a in filter(filter_arg, argv):
            self._given.add(a.lstrip('-'))

        # TODO: Also save which args have been modified from default values (check after ?)

        return self.args

    def save_args(self, config):
        """
        All arguments have to implement __str__. The program just try a
        naive conversion.
        Args:
            config (obj): configparser object
        """
        values = vars(self.args)
        for group_name, actions in self._group2key.items():
            if not actions:  # Skip positional and optional arguments
                continue

            config[group_name] = {}
            for a in actions:
                config[group_name][a] = str(values[a])

    def restore_args(self, config):
        """
        If a value cannot be fetched (ex: a new argument has been added), the
        default value will be used instead (None if not set).
        The arguments passed with the command line are not restored but kept
        Args:
            config (obj): configparser object
        """
        for group_name in config.sections():
            if group_name == ArgGroup.GLOBAL:  # Meta arguments are not restored
                continue

            for key in config[group_name]:
                action = self._key2action.get(key, None)
                if not action:  # Additionnal key is present (don't exist in argparse)
                    cprint(
                        'Warning: Could not restore param <{}/{}>. Ignoring...'.format(group_name,key),
                        color=TermMsg.WARNING
                    )
                    continue

                if key in self._given:  # The command lines arguments overwrite the given ones
                    continue

                # TODO: This code is critical so should be more carefully
                # tested (should also assert that the infered type correspond
                # to the expected one)
                # TODO: The paths are saved in their absolute form (models non
                # portable from a PC to another or if directory names changes)

                value = config[group_name][key]
                if action.type is not str or value == 'None':  # Use isinstance instead ?
                    value = ast.literal_eval(value)
                setattr(self.args, key, value)

            # If a key does not exist, the attribute won't be modified and
            # the default argparse value will be used (TODO: Should detect
            # and print warning if this was the case ?)

    def print_args(self):
        """ Print the registered arguments with their values.
        Need to be called after `parse_args`
        """
        values = vars(self.args)
        cprint('############### Parameters ###############', color=TermMsg.H1)
        for group_title, actions in self._group2key.items():
            if not actions:  # Skip positional and optional arguments
                continue
            cprint('[{}]'.format(group_title), color=TermMsg.H2)
            for a in actions:
                color = TermMsg.STRONG if a in self._given else None
                cprint('{}: {}'.format(a, values[a]), color=color)
            print()


class ArgDefault:  # TODO: Delete and dispatch among all modules
    """ Define the usuals arguments for machine learning research programs
    """

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.GLOBAL)
    def global_args(parser):
        parser.add_argument('--test', nargs='*', default=None, help='test mode')
        parser.add_argument('--model_tag', type=str, default=None, help='tag to differentiate which model to store/load')
        parser.add_argument('--dir_models', type=str, default=None, help='root folder in which the models are saved and loaded')
        parser.add_argument('--reset', action='store_true', help='use this if you want to ignore the previous model present on the model directory (Warning: the model will be destroyed with all the folder content)')

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.DATASET)
    def dataset_args(parser):
        parser.add_argument('--dataset', type=str, default=None, help='Dataset to use')
        parser.add_argument('--dataset_tag', type=str, default=None, help='tag to differentiate which dataset to store/load (dataset can have different hyperparameters too)')
        parser.add_argument('--dir_data', type=str, default=None, help='folder which contains the dataset')

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.TRAINING)
    def training_args(parser):
        parser.add_argument('--num_epochs', type=int, default=0, help='maximum number of epochs to run (0 for infinity)')
        parser.add_argument('--keep_every', type=float, default=0.3, help='if this option is set, a saved model will be keep every x hours (can be a fraction) (Warning: make sure you have enough free disk space or increase save_every)')
        parser.add_argument('--save_every', type=int, default=1000, help='nb of mini-batch step before creating a model checkpoint')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # TODO: Replace by more flexible module
        parser.add_argument('--batch_size', type=int, default=64, help='learning rate')
        parser.add_argument('--visualize_every', type=int, default=0, help='nb of mini-batch step before generating some outputs (allow to visualize the training)')
        parser.add_argument('--testing_curve', type=int, default=10, help='Also record the testing curve each every x iteration (given by the parameter)')
