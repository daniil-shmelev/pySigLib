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

"""Summarize paired timings, memory, working-set updates and numerical audits."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_search_policy import CASES, POLICIES


def read(path):
    return [json.loads(line) for line in (path / 'fits.jsonl').read_text().splitlines()]


def write_csv(path, rows):
    with path.open('w', newline='', encoding='ascii') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(out):
    records = read(out)
    assert all(r['ok'] for r in records)
    audits = {(r['policy'], r['case']): r for r in json.loads((out/'audit.json').read_text())}
    policies = list(dict.fromkeys(r['policy'] for r in records))
    rows = []
    for case in CASES:
        for policy in policies:
            group = sorted([r for r in records if r['case']==case and r['policy']==policy], key=lambda r:r['rep'])
            first, *timed = group
            assert first['fresh'] and first['rep']==-1 and [r['rep'] for r in timed]==list(range(7))
            t = np.array([r['seconds'] for r in timed])
            q1, mid, q3 = np.quantile(t,[.25,.5,.75])
            row = dict(case=case,policy=policy,median_s=mid,q1_s=q1,q3_s=q3,iqr_s=q3-q1,
                       native_mib=np.median([r['native_mib'] for r in timed]),rss_increase_mib=first['rss_mib'],
                       process_peak_mib=first['peak_mib'],correct=first['correct'],n=first['n'],support=first['support'],
                       nnz=first['nnz'],updates=np.median([r['updates'] for r in timed]),
                       passes=np.median([r['passes'] for r in timed]),activations=np.median([r['activated'] for r in timed]),
                       search_s=np.median([r['search_s'] for r in timed]),solve_s=np.median([r['solve_s'] for r in timed]),
                       certificate_s=np.median([r['cert_s'] for r in timed]),max_reported_kkt=max(r['kkt'] for r in group),
                       max_reported_gap=max(r['gap'] for r in group))
            row.update({k:v for k,v in audits[policy,case].items() if k in (
                'labels_changed','changed_words','changed_coefficients','objective_delta','test_score_delta','paper_correct','paper_words',
                'audited_words','exhaustive_feature_kkt','active_kkt','intercept_kkt')})
            row.setdefault('audited_words','')
            row.setdefault('exhaustive_feature_kkt','')
            rows.append(row)
    write_csv(out/'summary.csv',rows)
    data = {(r['case'],r['policy']):r for r in rows}
    totals = []
    for prefix in ('arabic','insect','speech','classifications','dict_','all'):
        cases = [c for c in CASES if (prefix=='all' or (prefix=='classifications' and not c.startswith('dict_')) or c.startswith(prefix))]
        for p in policies:
            group = [data[c,p] for c in cases]
            current_time = sum(data[c,'current']['median_s'] for c in cases)
            sec = sum(r['median_s'] for r in group)
            paired = [sum(r['seconds'] for r in records if r['policy']==p and r['case'] in cases and r['rep']==i) /
                      sum(r['seconds'] for r in records if r['policy']=='current' and r['case'] in cases and r['rep']==i) for i in range(7)]
            totals.append(dict(suite=prefix,policy=p,seconds=sec,ratio=sec/current_time,
                               paired_ratio_q1=float(np.quantile(paired,.25)),paired_ratio_q3=float(np.quantile(paired,.75)),
                               max_native_mib=max(r['native_mib'] for r in group),max_rss_mib=max(r['rss_increase_mib'] for r in group),
                               updates=sum(r['updates'] for r in group),passes=sum(r['passes'] for r in group),
                               faster=sum(r['median_s']<data[r['case'],'current']['median_s'] for r in group)))
    write_csv(out/'totals.csv',totals)
    traces = [r for r in records if r['policy']=='legacytrace' and r['rep']==-1]
    if traces:
        fields = ['pool','shared_support','peer','tree','raw_top_slots','adjacent_ties','batches_with_ties','candidates']
        write_csv(out/'ties_by_case.csv',[dict(case=r['case'],**dict(zip(fields,r['counts']))) for r in traces])
    pilot = []
    for folder in ('pilot','pool_pilot','peer_pilot','extremes','widths'):
        if not (out.parent/folder/'fits.jsonl').exists():
            continue
        results = read(out.parent/folder)
        for p in POLICIES:
            timed = [r for r in results if r['policy']==p and r['rep']>=0]
            if not timed:
                continue
            cases = list(dict.fromkeys(r['case'] for r in timed))
            sec = sum(float(np.median([r['seconds'] for r in timed if r['case']==c])) for c in cases)
            pilot.append(dict(panel=folder,policy=p,cases=','.join(cases),sum_medians_s=sec,
                              updates=sum(np.median([r['updates'] for r in timed if r['case']==c]) for c in cases),
                              passes=sum(np.median([r['passes'] for r in timed if r['case']==c]) for c in cases),
                              largest_native_mib=max(r['native_mib'] for r in timed),
                              failed=sum(not r['ok'] for r in timed)))
    write_csv(out/'pilot_summary.csv',pilot)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output)
