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
"""

import sys
import contextlib
import collections
import tqdm


# TODO: Utility to add color in terminal


__all__ = ['tqdm_redirect', 'OrderedAttr']


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
