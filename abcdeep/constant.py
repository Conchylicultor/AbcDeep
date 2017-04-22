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

""" Define standard constants and parameters to define the graphs
"""

# TODO: Move GraphKey to subgraph.py or rename constant.py => ... (Choose better name)


class GraphMode:
    """ Define the usual modes for the model
    Can be used at multiple places: At graph construction, at graph inference,
    for the data loader,...
    """
    TRAIN = 'train'
    VAL = 'val'  # Used while training
    TEST = 'test'  # Evaluate the final performance on the trained model
    ONLINE = 'online'  # For predicting a single sample/batch
    # TODO: Should the next two modes be merged somehow into the ONLINE case ?
    DAEMON = 'daemon'  #
    INTERACTIVE = 'interactive'  #


class GraphKey:
    """ Keys points in the graph for easy access to graph nodes
    """
    # TODO: Should separate the enum from the functions ?

    INPUT = 'input'  # Formatted input batch (after pre-processing) of the network
    TARGET = 'target'  # Used for training and metrics utilisation
    OUTPUT = 'output'
    LOSS = 'loss'
    IS_TRAIN = 'is_train'  # Access to a bool tensor (dropout or BN which behave
    # differently when training)
    OPTIMIZER = 'op_optimizer'
    OP_SUMMARY = 'op_summary'  # TODO: Not added here (the summary create its
    # own GraphKey to handle different priority level

    # Some hyperparameters
    LEARNING_RATE = 'learning_rate'

    _KEYS = {}  # TODO: Avoid using global cst (Allow to use 2 same network simultaneously) ?


    @staticmethod
    def add_key(key, value):
        """ Register the given graph node under a new key
        Args:
            key (str):
            value (Obj): the graph node (tensor, operator, placeholder,...)
        Raise:
            ValueError: if the given key has already been added previously
        """
        if key in GraphKey._KEYS:  # Check that the key is used only once (avoid collisions)
            raise ValueError('Error: GraphKey {} already set previously'.format(key))
        GraphKey._KEYS[key] = value
        # TODO: Modify graph collections to save the key with the graph ? (restore the keys)
        # TODO: Scope the keys under the graph_scope (allow to have same key under different graphs)


    @staticmethod
    def get_key(key):
        """
        Args:
            key (str):
        """
        return GraphKey._KEYS[key]
