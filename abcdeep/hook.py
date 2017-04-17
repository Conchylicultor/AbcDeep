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

""" Define the session run hooks main structure
"""

import tensorflow as tf


class AbcHook(tf.train.SessionRunHook):
    """ Base class for the hooks
    The hooks are similar to tf.train.SessionRunHook with some minor changes:
      * More methods are available to run special actions at specific times (ex:
      when the program exit)
      * The hooks can chontrol when it is executed (for which mode (train/val),
      every x iterations, every x seconds)
    """

    def __init__(self):
        self.glob_step = 0  # Each hook has its own copy of glob_step (TODO: Looks like a HACK to me)
        # TODO: The hook should have access to:
        #  * glob_step
        #  * current GraphMode


class PrintLossHook(AbcHook):
    """
    """
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(
            fetches={GraphKey.LOSS: GraphKey.get_key(GraphKey.LOSS)}
        )

    def after_run(self, run_context, run_values):
        print('Loss (iter {}): {} ()')
