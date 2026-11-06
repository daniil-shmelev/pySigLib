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

"""Summarize the split-screening measurements."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--verification', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('tmp/prefix_suffix_dispatch/screening_winners.json'))
    args = parser.parse_args()
    verified = {}
    for path in args.verification or []:
        for entry in json.loads(path.read_text()):
            verified[(entry['device'], entry['dtype'], tuple(entry['shape']), entry['phase'])] = entry
    entries = []
    for path in args.reports:
        report = json.loads(path.read_text())
        if not report.get('complete'): raise ValueError(f'Incomplete report: {path}')
        for case in report['cases']:
            for phase, result in case['confirmed'].items():
                if not result['accepted']: continue
                verification = verified.get((report['device'], report['dtype'], tuple(case['shape']), phase))
                if args.verification and (not verification or not verification['accepted']): continue
                entries.append(dict(report=str(path), device=report['device'], dtype=report['dtype'],
                                    shape=case['shape'], phase=phase, verification=verification, **result))
    args.output.write_text(json.dumps(entries, indent=2), encoding='ascii', newline='\r\n')
    for entry in entries:
        print(entry['device'], entry['dtype'], entry['shape'], entry['phase'],
              entry['candidate'], f"{entry['speedup']:.3f}x", entry['ci95'])
    print(f'{len(entries)} accepted choices')


if __name__ == '__main__':
    main()
