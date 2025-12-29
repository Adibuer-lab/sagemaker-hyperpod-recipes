# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import sys

from . import constants
from ..paths import get_launcher_scripts_path

_launcher_scripts_path = str(get_launcher_scripts_path())
if _launcher_scripts_path not in sys.path:
    sys.path.insert(0, _launcher_scripts_path)

try:
    from . import launchers, recipe_stages, slurm_launcher, stages
except ModuleNotFoundError:
    print("[WARNING] import launcher fail, is nemo_launcher available?")
