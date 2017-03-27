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

""" Other utilities. Contains:
 * tqdm_redirect: utility to redirect print to tqdm.write
 * OrderedAttr: keep track of the attribute declaration order in ._attr_values
 * cprint and TermMsg: collored version of print
"""

import sys
import contextlib
import collections
import tqdm


# TODO: Utility to add color in terminal


__all__ = ['tqdm_redirect', 'OrderedAttr', 'cprint', 'TermMsg']


class TqdmFile:
    """ Dummy file-like that will write to tqdm
    """
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def redirect_to_tqdm():
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    try:
        # Overwrite stdout and stderr
        sys.stdout = TqdmFile(sys.stdout)
        sys.stderr = sys.stdout

        yield save_stdout
    # Relay exceptions
    except Exception as exc:  # TODO: Is it necessary (not automatic) ?
        raise exc
    # Always restore sys.stdout if necessary
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


def tqdm_redirect(*args, **kwargs):
    """ Convinience wrapper to redirect stdout to tqdm
    """
    with redirect_to_tqdm() as f:
        # dynamic_ncols must be set because of a bug from tqdm side
        for x in tqdm.tqdm(*args, **kwargs, file=f, dynamic_ncols=True):
            yield x


class OrderedAttrMeta(type):
    """ Replace the standard __dict__ attribute by an OrderedDict
    This allows to keep tracks of the attribute declaration order
    """
    @classmethod
    def __prepare__(metacls, name, bases):
        """ Overload the default __dict__
        """
        return collections.OrderedDict()

    def __new__(cls, name, bases, classdict):
        """ When the class is created, the attributes vales are saved with their
        order
        """
        result = type.__new__(cls, name, bases, dict(classdict))
        exclude = set(dir(type))  # TODO: Also filter functions
        attr_values = [v for k, v in classdict.items() if k not in exclude]
        result._attr_values = attr_values
        return result


class OrderedAttr(metaclass=OrderedAttrMeta):
    """ Convinience class to hidde the metaclass
    """
    def __setattr__(self, name, value):  # TODO: Useless ?
        """ Called when a new attribute is added.
        Update the ordered keys
        """
        if not hasattr(self, name):
            self._attr_values.append(value)
        return super().__setattr__(name, value)


class TermColorMsg:
    """ Enumerate ASCII color messages
    """
    TXT_BLACK = '\033[30m'
    TXT_RED = '\033[31m'
    TXT_GREEN = '\033[32m'
    TXT_YELLOW = '\033[33m'  # The 93m values gives other color
    TXT_BLUE = '\033[34m'
    TXT_PINK = '\033[35m'
    TXT_CYAN = '\033[36m'
    TXT_WHITE = '\033[37m'

    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_PINK = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

    END = '\033[0m'


class TermMsg:
    """ Color messages, sementic version
    """
    H1 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_WHITE
    H2 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_CYAN
    H3 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_PINK
    STRONG = TermColorMsg.BOLD
    EM = TermColorMsg.ITALIC
    SUCCESS = TermColorMsg.TXT_GREEN
    FAIL = TermColorMsg.BOLD + TermColorMsg.TXT_RED
    WARNING = TermColorMsg.TXT_BLACK + TermColorMsg.BG_YELLOW


def cprint(*arg, color=None, **kwargs):
    """ Print the given arguments with the given color
    """
    assert color is not None
    # end = kwargs.get('end', None)  # TODO: Allow to choose the end
    print(color, end='')
    print(*arg, **kwargs, end='')
    print(TermColorMsg.END, end='')
    print()
