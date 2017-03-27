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

""" File system management and string utilities
"""

import os
import collections


def get_filelist(dirname, ext=None):
    """ Return all files in a directory sorted
    Args:
        dirname (str)
        ext (str or list[str]): the file extentions to keep
    Return:
        list[str]: the filenames (don't contains the dirname)
    Raise:
        ValueError: if the returned list is empty
    """
    if ext is not None and not isinstance(ext, collections.Iterable):
        ext = [ext]

    def is_file_valid(filename):
        return (
            os.path.isfile(os.path.join(dirname, filename)) and  # Exclude dir
            (ext is None or any(filename.endswith(e) for e in ext))
        )

    filenames = list(sorted(filter(is_image_valid, os.listdir(dirname))))

    if not len(filenames):
        raise ValueError('{} does not contains any valid images'.format(dirname))

    return filenames


def rstrip(s, suffix):
    """ Return the given string without the suffix at the end
    """
    if s and s.endswith(suffix):
        return s[:-len(suffix)]
    return s
