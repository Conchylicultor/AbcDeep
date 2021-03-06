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

"""
"""

import tensorflow as tf


def create_queue(x, y, batch_size):
    """ Temporary function which create a queue from the given data
    Args:
        x: Input
        y: Target
    Return:
        (t_x, t_y): a tuple of tensor corresponding to the queue output
    """
    # TODO: Replace this function by more advanced/flexible queue
    return tf.train.batch([x, y], batch_size)


class InputQueues:
    pass
