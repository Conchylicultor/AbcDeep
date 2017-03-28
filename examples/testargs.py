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

""" Demonstration of the argparse extention
"""

import os
import configparser

from abcdeep.argsutils import ArgParser, ArgGroup, ArgDefault
from abcdeep.otherutils import cprint, TermMsg


class OtherArgs:
    @staticmethod
    @ArgParser.regiser_args(ArgGroup.SIMULATOR)
    def simulator_args(parser):
        parser.add_argument('--physics', action='store_true', help='Activate the physics engine')
        parser.add_argument('--record', action='store_true', help='Record the game')

    @staticmethod
    @ArgParser.regiser_args('Custom group')
    def custom_args(parser):
        parser.add_argument('--first', type=str, default=None, help='First arg')
        parser.add_argument('--last', type=str, default=None, help='Last arg')
        parser.add_argument('--choices', type=str, choices=['C1', 'C2', 'C3'], default='C2', help='Last arg')

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.GLOBAL)
    def previous_args(parser):
        parser.add_argument('--mode', choices=[1, 2, 3], default=None, help='First arg')


def create_argparser():
    argparse = ArgParser()  # Create the parser

    argparse.register_cls(ArgDefault)  # Add some arguments
    argparse.register_cls(OtherArgs)

    argparse.overwrite(  # Change default arguments values
        batch_size=1024,
        last='this has been overwritten',
    )

    return argparse


def test_args():
    argparse1 = create_argparser()
    argparse1.parse_args()  # Parse the default command line
    argparse1.print_args()

    config_name = 'examples/config.ini'  # TODO: Dynamically compute the path

    # Saving args
    cprint('WARNING: Saving args...', color=TermMsg.WARNING)
    config = configparser.ConfigParser()
    argparse1.save_args(config)
    with open(config_name, 'w') as config_file:
        config.write(config_file)
    with open(config_name) as f:
        print(f.read())

    # Restoring args
    argparse2 = create_argparser()
    argparse2.parse_args(['--batch_size', '256'])  # Manually overwrite default arguments

    cprint('WARNING: Restoring args...', color=TermMsg.WARNING)
    config = configparser.ConfigParser()
    config.read(config_name)
    argparse2.restore_args(config)
    argparse2.print_args()

    os.remove(config_name)  # Cleanup
