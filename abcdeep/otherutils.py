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
 * gen_attr: recursivelly generate the list of all attribute of a given object
 * iterify: Ecnapsulate a given object into a list
 * tqdm_redirect: utility to redirect print to tqdm.write
 * interrupt_handler: Context manager to capture signal.SIGINT
 * OrderedAttr: keep track of the attribute declaration order in ._attr_values
 * cprint and TermMsg: collored version of print
"""

import sys
import signal  # For capturing Ctrl+C
import inspect  # To check if isclass
import contextlib
import collections
import tqdm


__all__ = [
    'gen_attr',
    'iterify',
    'tqdm_redirect',
    'tqdm_redirector',
    'interrupt_handler',
    'OrderedAttr',
    'cprint',
    'TermMsg'
]


def gen_attr(obj):
    """ Generate the list of all members of a class
    Include those in parent classes
    Args:
        obj: class or instance
    Return:
        attr: yield all attribute values
    """
    isclass = inspect.isclass(obj)
    cls = obj if isclass else type(obj)
    base = () if isclass else (obj,)

    for c in reversed(base + cls.__mro__):  # Reversed because top level classes need to be parsed first (for args order)
        for v in vars(c):
            yield getattr(obj, v)


def iterify(obj):
    """ Encapsulate object in an iterable (list) if necessary
    Ex:
        a = iterify(None)  # a = []
        a = iterify(obj)  # a = [obj]
        a = iterify([1, 2, 3])  # a = [1, 2, 3]

    Args:
        obj (List[T], None or T): Object to iterify
    Return:
        List[T]: The encapsulated object, empty list if obj is None
    """
    if obj is None:  # None -> []
        obj = []
    if not isinstance(obj, collections.Iterable):  # obj -> [obj]
        obj = [obj]
    return obj


class TqdmFile:
    """ Dummy file-like that will write to tqdm
    """
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file)


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


def tqdm_redirector(*args, **kwargs):
    """ Convinience wrapper to redirect stdout to tqdm.
    Yield the tqdm object the first time the function is called to allow custom
    control over it.
    Close the tqdm object the second time the function is called
    """
    # TODO: Replace this function by an object which generate tqdm objects on
    # the fly. Every time a new object is created, the previous one is closed
    with redirect_to_tqdm() as f:
        # dynamic_ncols must be set because of a bug from tqdm side
        t = tqdm.tqdm(*args, **kwargs, file=f, dynamic_ncols=True)
        yield t
        t.close()


@contextlib.contextmanager
def interrupt_handler():
    """ Context manager to capture the Ctr+C interruption
    Usage:

        with interrupt_handler() as h:
            while True:
                if h.interrupted:  # Ctrl+C pressed
                    break
    """
    sig = signal.SIGINT
    # TODO: Also capture SIGTERM (ex: when using pycharm) ?

    class InterruptState:
        def __init__(self):
            self.interrupted = False

        def handler(self, signum, frame):
            self.interrupted = True

    try:
        state = InterruptState()
        prev_handler = signal.signal(sig, state.handler)
        yield state
    finally:
        signal.signal(sig, prev_handler)  # Restore original signal handler


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
    """ Convinience class to hide the metaclass
    """
    def __setattr__(self, name, value):  # TODO: Useless ? Should also update when delete
        """ Called when a new attribute is added.
        Update the ordered keys
        """
        if not hasattr(self, name):
            self._attr_values.append(value)
        return super().__setattr__(name, value)

    @classmethod
    def arg_choices(cls):
        """ Helper function to generate the arguments for argparse
        Is used with **. For example:

            parser.add_argument(
                '--mode',
                **Mode.arg_choices(),
                help='program mode'
            )

        Warning: Will only work if the class is an enum of string
        """
        return {
            'choices': cls._attr_values,
            'default': cls._attr_values[0],
            'type': str,
        }


# TODO: Create a mapping class (enum to object, for instance:
# InputType.MIDI -> 'midi' -> MidiConnector) ? Would allows easy conversions
# between argparse choices and other objects


class TermColorMsg:
    """ Enumerate ASCII color messages
    """
    END = '\033[0m'
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    SELECTED = '\033[7m'  # Text and background color inverted

    TXT_BLACK = '\033[30m'
    TXT_RED = '\033[31m'
    TXT_GREEN = '\033[32m'
    TXT_YELLOW = '\033[33m'
    TXT_BLUE = '\033[34m'
    TXT_MAGENTA = '\033[35m'
    TXT_CYAN = '\033[36m'
    TXT_WHITE = '\033[37m'

    TXT_GREY    = '\033[90m'
    TXT_RED2    = '\033[91m'
    TXT_GREEN2  = '\033[92m'
    TXT_YELLOW2 = '\033[93m'
    TXT_BLUE2   = '\033[94m'
    TXT_MAGENTA2 = '\033[95m'
    TXT_CYAN2  = '\033[96m'
    TXT_WHITE2  = '\033[97m'

    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

    BG_GREY    = '\033[100m'
    BG_RED2    = '\033[101m'
    BG_GREEN2  = '\033[102m'
    BG_YELLOW2 = '\033[103m'
    BG_BLUE2   = '\033[104m'
    BG_MAGENTA2 = '\033[105m'
    BG_CYAN2  = '\033[106m'
    BG_WHITE2  = '\033[107m'


class TermMsg:
    """ Color messages, semantic version
    """
    H1 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_WHITE
    H2 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_CYAN
    H3 = TermColorMsg.TXT_BLACK + TermColorMsg.BG_MAGENTA
    STRONG = TermColorMsg.BOLD
    EM = TermColorMsg.ITALIC
    SUCCESS = TermColorMsg.TXT_GREEN
    FAIL = TermColorMsg.BOLD + TermColorMsg.TXT_RED
    WARNING = TermColorMsg.TXT_BLACK + TermColorMsg.BG_YELLOW


def cprint(text, color=None, **kwargs):
    """ Print the given arguments with the given color
    """
    if color is None:
        print(text, **kwargs)
    else:
        print(color + text + TermColorMsg.END, **kwargs)
