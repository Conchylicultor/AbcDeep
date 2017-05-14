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

""" Define the class to control when the hooks are called
"""

import functools

import abcdeep
import abcdeep.otherutils as otherutils
from abcdeep.constant import GraphMode


class HookController:
    """ Base Controller Class which determine when the hooks are triggered
    The class itself don't do anything but is meant to be subclassed
    """
    FCT_FLAG = '_hookcontroler_flag'

    @staticmethod
    def overwrite(f):
        """ Decorator which flag the function as overwriting the hook one
        The functions flagged decorate the hook
        """
        setattr(f, HookController.FCT_FLAG, True)
        return f

    def __init__(self):
        class Namespace: pass

        self.hook = None  # Reference on the controled hook
        self.state = None
        self.fct = Namespace()  # Keeps reference on overwritten functions

    def apply(self, hook):
        """ Calling this function will decorate the given hook
        """
        self.hook = hook
        self.state = hook.state

        for fct_ctrl in otherutils.gen_attr(self):
            if hasattr(fct_ctrl, HookController.FCT_FLAG):
                name = fct_ctrl.__name__
                fct_hook = getattr(hook, name)
                assert fct_hook  # Make sure the name we overwritten is correct

                # TODO: Use functools.update_wrapper ?
                setattr(self.hook, name, fct_ctrl)
                setattr(self.fct, name, fct_hook)


class OnlyModeController(HookController):
    """
    Only trigger for the given modes
    """
    def __init__(self, modes=None):
        super().__init__()
        self.modes = abcdeep.iterify(modes, default=[GraphMode.TRAIN])

    def activated(self):
        return self.state.curr_mode in self.modes

    @HookController.overwrite
    def before_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.before_run(*args, **kwargs)

    @HookController.overwrite
    def after_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.after_run(*args, **kwargs)


class EveryXIterController(HookController):
    def __init__(self, num_step, at_first=True):
        """
        Args:
            num_step (int):
            at_first (bool): True if save at first iteration
        """
        super().__init__()
        self.num_step = num_step
        self.at_first = at_first

    def activated(self):
        if not self.at_first and self.state.glob_step == 0:
            return False
        return self.state.glob_step % self.num_step == 0

    @HookController.overwrite
    def before_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.before_run(*args, **kwargs)

    @HookController.overwrite
    def after_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.after_run(*args, **kwargs)


class DefaultPolicy:
    """ Only train
    """
    def run(self, state):
        return GraphMode.TRAIN


class AlternatePolicy:
    """ Run train and val simultaneously
    """
    # TODO: Loss print once both train and test
    # TODO: Be carefull that hooks which use some val_every variable are run even if not
    # synchronous with AlternatePolicy.VAL_EVERY (ex: summary record val every 13 iter and
    # AlternatePolicy switch val mode every 10 iter)
    # TODO: Only when train mode is the optimizer step run (.forward() and .backward())
    # TODO: Summary train/test
    def __init__(self):
        self.prev = False  # Avoid running val twice at the same iteration
        self.VAL_EVERY = 10  # TODO: Avoid hardcoded value ?

        self.gen_next = None

    def run(self, state):
        def next_mode():
            while True:
                if state.glob_step % self.VAL_EVERY == 0:
                    yield GraphMode.VAL
                yield GraphMode.TRAIN

        if self.gen_next is None:
            self.gen_next = next_mode()
        return next(self.gen_next)
