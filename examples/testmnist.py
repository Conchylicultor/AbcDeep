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

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn

import abcdeep
import abcdeep.queue
from abcdeep.argsutils import ArgParser, ArgGroup
from abcdeep.abcprogram import AbcProgram
from abcdeep.abcsubgraph import AbcModel, AbcDataConnector
from abcdeep.constant import GraphMode, GraphKey
from abcdeep.summary import SummaryKeys


class MnistLoader(AbcDataConnector):
    """ Define the input queues for training/validation/testing
    """
    @staticmethod
    @ArgParser.regiser_args(ArgGroup.NETWORK)
    def model_args(parser):
        parser.add_argument('--img_size', type=int, default=28, help='Input size, height and width of the image')

    @staticmethod
    @ArgParser.regiser_args(ArgGroup.TRAINING)
    def training_args(parser):
        # TODO: Move batch_size declaration elsewhere ?
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for a training/validation iteration')

    def _init(self, state):
        super()._init(state)
        #self._build_queue()
        self._build_graph()

    def _build_graph(self):
        self.mnist = learn.datasets.load_dataset('mnist')

        # TODO: Use build_queue instead
        img_size = self.state.args.img_size

        self.p_image = tf.placeholder(
            tf.float32,
            shape=(None, img_size*img_size,),
            name='image',
        )
        t_image = tf.reshape(self.p_image, [-1,img_size,img_size,1])

        self.p_target = tf.placeholder(
            tf.int32,
            shape=(None),
            name='target',
        )
        GraphKey.add_key(GraphKey.INPUT, t_image)
        GraphKey.add_key(GraphKey.TARGET, self.p_target)

    def _build_queue(self):
        # TODO: Does not seem possible to control the dataset saving location
        mnist = learn.datasets.load_dataset('mnist')

        img_size = self.state.args.img_size
        batch_size = self.state.args.batch_size

        t_image, t_target = tf.train.batch(
            [mnist.train.images, mnist.train.labels],
            batch_size=batch_size,
            enqueue_many=True,
        )

        # TODO: Add data augmentation
        # TODO: Use DataHook instead (avoid duplicating code if multiple queues ?)
        t_image = tf.reshape(t_image, [-1,img_size,img_size,1])
        t_target = tf.cast(t_target, tf.int32)

        GraphKey.add_key(GraphKey.INPUT, t_image)
        GraphKey.add_key(GraphKey.TARGET, t_target)


    def _build_queue_proto(self):
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

    def before_run(self, run_context):
        """
        """
        images, targets = self.mnist.train.next_batch(self.state.args.batch_size)
        return tf.train.SessionRunArgs(None, feed_dict={
            self.p_image: images,
            self.p_target: targets,
        })


class Model(AbcModel):
    """ Define the network architecture as well as the optimizer
    """
    @staticmethod
    @ArgParser.regiser_args(ArgGroup.NETWORK)
    def model_args(parser):
        parser.add_argument('--nb_class', type=int, default=10, help='Output size, number of classes to predict')

    def _init(self, state):
        super()._init(state)

        self._build_network()
        self._build_loss()
        self._build_optimizer()  # TODO: Only build if not training


    def _build_network(self):
        with tf.variable_scope('network'):  # TODO: Use decorator instead as
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
            net = tf.layers.dense(inputs=net, units=self.state.args.nb_class)
            GraphKey.add_key(GraphKey.OUTPUT, net)

    def _build_loss(self):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=GraphKey.get_key(GraphKey.OUTPUT),
                labels=GraphKey.get_key(GraphKey.TARGET),
                # dim=-1,  # Don't exist for sparse_s...
            ))

            SummaryKeys.add_scalar('loss', loss)
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

    def before_run(self, run_context):
        """
        """
        # TODO: Could have a hook which automatically run a train step if training mode
        return tf.train.SessionRunArgs(
            fetches={GraphKey.OPTIMIZER: GraphKey.get_key(GraphKey.OPTIMIZER)},
        )


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
    program.run()
