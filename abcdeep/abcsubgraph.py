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

import abcdeep
from abcdeep.hook import AbcHook


# TODO: Does every SubGraph also are hooks (modify the graph when necessary) ?

class SubGraph:
    def forward():
        """
        Each
        * For the dataloader, generate and feed the next batch.
        * For the model, add the operators to run for the current loop
        * For the summary, add the summaries to record for this loop
        """
        pass

    def backward():
        pass


class ModeSelector(SubGraph):
    """
    """

    def add_mode():
        pass

    def set_mode(self, new_mode):
        pass


class AbcModel(SubGraph, AbcHook):  # TODO: It's SubGraph which should inherit from AbcHook
    """
    A model is also a hook and can control which operations are executed on the
    graph
    """
    # TODO: Use separate class ForwardHook, BackwardHook instead ?


class AbcDataConnector(SubGraph, AbcHook):
    """
    Is used both as a Hook (load data at each iterations) and as a gaph constructor
    TODO: Better to use 2 different class: One to create the graph and the second
    which clall DataConnector.next_batch_train() ?
    """

    def get_epoch_length(self):
        """
        Return:
            int: the current epoch length
        """
        pass
