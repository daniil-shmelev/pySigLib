# Copyright 2025 Daniil Shmelev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import numpy as np
import iisignature

import pysiglib

if __name__ == '__main__':
    X = np.random.randint(low=0, high=10, size=(1000, 2)).astype("double")

    iisigres = iisignature.sig(X, 3)
    myres = pysiglib.sig(X, 3)

    print(iisigres)
    print(myres)

    for i in range(5):
        print("#" * 30)

    X = np.random.randint(low = 0, high = 10, size = (5, 10, 2)).astype("double")

    iisigres = iisignature.sig(X, 3)
    myres = pysiglib.sig(X, 3)

    for i in range(5):
        print(iisigres[i])
        print(myres[i])
        print("#" * 30)
