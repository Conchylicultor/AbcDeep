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

""" Simple digit classification with CNN
"""

import tensorflow as tf
import tensorflow.contrib.learn as learn

import abcdeep
from abcdeep.argsutils import ArgParser, ArgGroup
from abcdeep.abcprogram import AbcProgram
from abcdeep.abcsubgraph import AbcModel
from abcdeep.constant import GraphMode, GraphKey


class MnistLoader():
    """ Define the input queues for training/validation/testing
    """
    def build_queue(self):
        mnist = learn.datasets.load_dataset('mnist')

        inputs = abcdeep.queue.InputQueues()
        inputs.add_queues(
            # key='',
            collections=[GraphMode.TRAIN, GraphMode.VAL, GraphMode.TEST],
            hooks=[abcdeep.queue.BatchSummary()],
            data=[mnist.train, mnist.validation, mnist.test]
        )

        outputs = inputs.get_outputs()


        inputs.add_queue(
            key='',
            collections=GraphMode.VAL,
            hooks=[abcdeep.queue.BatchSummary()],
            data=mnist.validation
        )
        inputs.add_queue(
            key='',
            collections=GraphMode.TEST,
            data=mnist.test
        )
        inputs.add_queue(
            key='',
            collections=GraphMode.ONLINE,
            data=tf.placeholder
        )

        outputs = inputs.get_outputs()


class Model(AbcModel):
    """ Define the network architecture as well as the optimizer
    """
    @staticmethod
    @ArgParser.regiser_args(ArgGroup.NETWORK)
    def model_args(parser):
        pass

    def _build_network(self, input_img, args, p_train):
        with tf.name_scope('network'):  # TODO: Use decorator instead as
            # @abcdeep.scope() which would parse the _build_{name_scope} function
            # name ?
            net = GraphKey.get_key(GraphKey.INPUT)
            net = tf.layers.conv2d(
                inputs=net,
                filters=32,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu
            )
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
            net = tf.layers.conv2d(
                inputs=net,
                filters=64,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu
            )
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
            net = tf.layers.dropout(
                inputs=net,
                rate=0.4,
                training=GraphKey.get_key(GraphKey.IS_TRAIN)
            )
            net = tf.layers.dense(inputs=net, units=10)
            GraphKey.add_key(GraphKey.OUTPUT, net)

    def _build_loss(self):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=GraphKey.get_key(GraphKey.OUTPUT),
                labels=GraphKey.get_key(GraphKey.TARGET),
                # dim=-1,  # Don't exist for sparse_s...
            ))

            GraphKey.add_key(GraphKey.LOSS, loss)

        # TODO: tf.summary.scalar('loss', self.loss)  # Keep track of the cost

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=GraphKey.get_key(GraphKey.LEARNING_RATE),
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            # Try gradient post-processing (clipping, visualisation,...) ??

            # Also update BN moving average and variance
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                op_opt = opt.minimize(GraphKey.get_key(GraphKey.LOSS))
                GraphKey.add_key(GraphKey.OPTIMIZER, op_opt)


class Program(AbcProgram):
    def __init__(self):
        super().__init__(
            abcdeep.ProgramInfo('MnistClassif', '0.1'),
            dataconnector=MnistLoader,
            model=Model,
            #trainer=abcdeep.Trainer,
        )

        # TODO: Add all hooks

    def _add_modes(self):
        # TODO: Can the modes be inside __init__ (probably not because depend of args)

        # TODO: The first added mode is the one launched by default
        # TODO:

        self.add_mode(
            GraphMode.TRAIN,
            hooks=[AutoSaver(), PrintLoss()],
        )
        self.add_mode(
            GraphMode.VAL,
            hooks=[LoaderSaver(), PrintLoss()],
        )
        self.attach_mode(  # Run train and val simultaneously
            GraphMode.VAL,
            GraphMode.TRAIN,
            every_it=100,  # Run a validation step every x train iterations
        )

def test_mnist():
    program = Program()
    #    dataloader=MnistLoader,
    #    model=Model,
    #    trainer=abcdeep.Trainer,
    #    config=[],  # Also add saver/summary by default
    #)
    program.run()
