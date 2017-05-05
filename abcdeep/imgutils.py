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

""" Image preprocessing utilities
"""

import tensorflow as tf


def center_scale(img):
    # Finally, rescale to [-1,1] instead of [0, 1)
    with tf.name_scope('center_scale'):
        img = img - 0.5
        img = img * 2.0
    # The centered scaled image is directly compatible with tf.summary
    return img


def _img_transformation(self, img):
    """ Apply some random transformations to the tensor image.
    Use the same transformations that the official inception code
    Warning: The input image should range from [0-1]
    Return the new image
    """
    with tf.name_scope('image_transform'):
        # TODO: If modifying the orientation, also need to modify the labels accordingly
        #img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_flip_up_down(img)

        choice = tf.random_uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)  # Generate a number inside [0,1]
        choice = tf.cast(choice, tf.bool)

        brightness = functools.partial(tf.image.random_brightness, max_delta=32/255) # imgnet: 32. / 255. ? Ciffar: 63 TODO: Tune
        contrast = functools.partial(tf.image.random_contrast, lower=0.5, upper=1.5) # imgnet: lower=0.5, upper=1.5
        #hue = functools.partial(tf.image.random_hue, max_delta=0.2)
        saturation = functools.partial(tf.image.random_saturation, lower=0.5, upper=1.5)

        choices = [
            [brightness, saturation, contrast],
            [brightness, contrast, saturation],
        ]

        def transform(input_img, n=None):
            for fct in choices[n]:
                input_img = fct(input_img)
            return input_img

        # Randomly apply transform order 1 or 2
        transform1 = functools.partial(transform, img, n=0)
        transform2 = functools.partial(transform, img, n=1)
        img = tf.cond(choice, transform1, transform2)

        # The random_* ops do not necessarily clamp.
        img = tf.clip_by_value(img, 0.0, 1.0)

        return img


def get_t_jpg(filename, preprocessing=None):
    """
    Return a float tensor image in range [-1.0 ; 1.0]
    preprocessing is a transformation function
    Preprocessing is done with float32 images in range [0.0, 1.0]
    """
    # TODO: Difference between tf.read_file() and tf.WholeFileReader() ?
    # TODO: Add summary ?

    t_image = tf.read_file(filename)
    t_image = tf.image.decode_jpeg(t_image, channels=3)  # [0-255], RGB
    t_image = tf.image.convert_image_dtype(t_image, dtype=tf.float32)  # Range: [0-1]

    # Should normalize images ? How to compute mean-std on the
    # training set (randomly sample 10000 images from the training set
    # Would this bias the result ? Should we just perform a simple
    # linear scaling), use per_image_standardization ?
    # tf code for inception only call convert_image_dtype and at the end (
    # after image preprocessing, scale to [-1.0;1.0])

    if preprocessing is not None:
        t_image = preprocessing(t_image)

    t_image = center_scale(t_image)  # Finally, rescale to [-1,1] instead of [0, 1)

    return t_image
