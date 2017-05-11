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

import abcdeep.otherutils as otherutils


class Overwrited:
    """ Decorator for the overwritten function which call
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        pass



class HookController:
    """ Base Controller Class which determine when the hooks are triggered
    The class itself don't do anything but is meant to be subclassed
    """
    FCT_FLAG = '_hookcontroler_flag'

    @staticmethod
    def overwrite(f):
        """ Decorator which flag the function as overwriting the hook one
        The functions have to be of the form
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


class EveryXIterController(HookController):
    def __init__(self, num_step):
        super().__init__()
        self.num_step = num_step

    def activated(self):
        return self.state.glob_step % self.num_step == 0

    @HookController.overwrite
    def before_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.before_run(*args, **kwargs)

    @HookController.overwrite
    def after_run(self, *args, **kwargs):
        if self.activated():
            return self.fct.after_run(*args, **kwargs)
