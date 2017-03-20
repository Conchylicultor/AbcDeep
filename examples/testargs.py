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

from abcdeep.argsutils import ArgParser, ArgGroup, ArgDefault

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

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.GLOBAL)
    def previous_args(parser):
        parser.add_argument('--mode', choices=[1, 2, 3], default=None, help='First arg')


def test_args():
    argparse = ArgParser()  # Create the parser

    argparse.register_cls(ArgDefault)  # Add some arguments
    argparse.register_cls(OtherArgs)

    argparse.overwrite(  # Change default arguments values
        batch_size=1024,
        last='abcd'
    )

    argparse.parse_args()  # Parse the default command line

    argparse.print_args()
