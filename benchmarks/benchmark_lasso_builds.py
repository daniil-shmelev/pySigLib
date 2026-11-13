# Copyright 2026 Daniil Shmelev
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

"""Compare saved native builds in separate, ordered worker processes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/optimization_round3/final_fresh'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--reference', type=Path, default=Path('output/optimization_round2/measured_source/pysiglib/cpsig.dll'))
    parser.add_argument('--candidate', type=Path, default=Path('out/build/tt/siglib/cpsig/Release/cpsig.dll'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=3600)
    args = parser.parse_args()
    destination = Path('pysiglib/cpsig.dll')
    libraries = {'optimized': args.reference.resolve(), 'optimized_next': args.candidate.resolve()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in libraries.items()}
    if hashlib.sha256(destination.read_bytes()).hexdigest() != hashes['optimized_next']:
        raise RuntimeError('The loaded DLL does not match the candidate build')
    try:
        for case_index, case in enumerate(args.cases):
            folder = args.output/case
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'spec.json').exists():
                raise FileExistsError(folder/'spec.json')
            shutil.copy2(args.data/case/'data.npz', folder/'data.npz')
            spec = json.loads((args.data/case/'spec.json').read_text())
            spec.update(repeats=args.repeats, warmups=args.warmups, reuse_process=False,
                        methods=list(libraries), native_builds=hashes, comparison='saved native builds')
            for repeat in range(-args.warmups, args.repeats):
                methods = list(libraries)
                if (repeat+case_index) % 2: methods.reverse()
                for method in methods:
                    shutil.copy2(libraries[method], destination)
                    spec['native_sha256'] = hashes[method]
                    (folder/'spec.json').write_text(json.dumps(spec, indent=2)+'\n')
                    try:
                        subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_lasso_optimization', '--worker', str(folder.resolve()),
                                        '--method', method, '--repeat', str(repeat)], check=True, timeout=args.timeout)
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                        (folder/'failure.json').write_text(json.dumps(dict(method=method, repeat=repeat, error=str(error)), indent=2)+'\n')
                        raise
    finally:
        shutil.copy2(libraries['optimized_next'], destination)


if __name__ == '__main__':
    main()
