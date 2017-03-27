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

""" Demonstration of the color extention
"""

from abcdeep.otherutils import cprint, TermColorMsg, TermMsg


def print_colors(cls):
    """ Print all available messages with their associated color
    """
    for m in sorted(vars(cls)):
        if m.startswith('_'):
            continue
        cprint(m, color=getattr(cls, m))


def print_format_table():
    """ prints table of formatted text format options
    From stackoverflow
    """
    for style in range(8):
        for fg in range(30,38):
            s1 = ''
            for bg in range(40,48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')

def test_colors():
    cprint('########## Colors messages ##########', color=TermMsg.H1)
    print_colors(TermColorMsg)
    cprint('########## Semantic messages ##########', color=TermMsg.H1)
    print_colors(TermMsg)
    #cprint('########## All colors ##########', color=TermMsg.H1)
    #print_format_table()
