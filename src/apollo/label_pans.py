"""Turn parsed ALSJ pans into a station-labelled place-recognition benchmark.

Keeps ordinary 60-mm surface pans taken from one standpoint, assigns each a station label
(mission + station), and drops telephoto (500-mm) series, traverse/drive sequences, window
pans, detail crops, composites and duplicate re-assemblies.
Writes pans_labelled.json and benchmark.json.
"""
import json
import re
from collections import defaultdict

DROP = re.compile(r'500-?mm|500mm|Traverse|Drive|Return|Animation|Film Strip|Window|Portrait|'
                  r'High[- ]Res|Detail|Composite|Anaglyph|Polarization|Closeout|Partial|Geophone|'
                  r'LRV Pan|Seatbelt|Eastern Horizon|View to the|B&W of LM|Running|UV Camera|'
                  r'Outcrop|Massif|Boulder Tracks|Diabase|Stereo|Shadow Rock|Crater Detail|'
                  r'Non-Crater|Ballet|^\s*$|^\.|^and |^Northern', re.I)


def station(m, name):
    n = name
    s = re.search(r'Station\s+([0-9]+[a-zA-Z]?(?:-Prime)?|[A-G](?:-?Prime|-?\d)?)', n)
    if s:
        return s.group(1).replace('-', '').replace('Prime', "'")
    if re.search(r'ALSEP', n):
        return 'ALSEP'
    if m == 11 and re.match(r'Pan \d|Ladder', n):
        return 'LM'
    if re.search(r"O'?clock|LM Pan|LM pan|Plus-Z|EVA-1 Closeout", n, re.I):
        return 'LM'
    if re.search(r'SEVA', n):
        return 'SEVA'
    if re.search(r'Middle Crescent', n):
        return 'MiddleCrescent'
    if re.search(r'Victory', n):
        return 'Victory'
    if re.search(r'Buster', n):
        return '2'
    return None


out = []
for m in [11, 12, 14, 15, 16, 17]:
    seen = set()
    for i, p in enumerate(json.load(open(f'pans{m}.json'))):
        name = p['name']
        key = (p['frames'][0], len(p['frames']))
        if key in seen:
            continue
        seen.add(key)
        if p['get'] is None:  # name swallowed the description; take its leading words
            name = re.sub(r'^\d{3}:\d{2}:\d{2}\s*', '', name)
        if DROP.search(name):
            continue
        st = station(m, name)
        if st is None or len(p['frames']) < 4:
            continue
        out.append({'mission': m, 'idx': i, 'name': name[:80], 'station': f'A{m}-{st}',
                    'get': p['get'], 'frames': p['frames']})

# a frame may appear in two pans (re-assemblies); keep first
owner = {}
for k, p in enumerate(out):
    p['frames'] = [f for f in p['frames'] if owner.setdefault(f, k) == k]
json.dump(out, open('pans_labelled.json', 'w'), indent=1)

by_st = defaultdict(list)
for k, p in enumerate(out):
    by_st[p['station']].append(k)
multi = {s: v for s, v in by_st.items() if len(v) >= 2}
print(f'{len(out)} pans, {sum(len(p["frames"]) for p in out)} frames, {len(by_st)} stations, '
      f'{len(multi)} stations with >=2 pans')
for s, v in sorted(by_st.items()):
    print(f'  {s:18s}', ' | '.join(f'{out[k]["name"][:34]} ({len(out[k]["frames"])})' for k in v))
