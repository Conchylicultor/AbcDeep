#!/usr/bin/env python3

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

import examples.testargs
import examples.testcolors
import examples.testmnist


# TODO: Use abcdeep.ArgParser to choose which demo run


if __name__ == '__main__':
    #examples.testcolors.test_colors()
    #examples.testargs.test_args()
    examples.testmnist.test_mnist()
