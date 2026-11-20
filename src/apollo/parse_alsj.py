"""Parse ALSJ image-library pages into per-frame captions, station labels and pan membership.

Usage: python parse_alsj.py alsj11.html ... > frames.jsonl  (writes pans.json too)
"""
import html as H
import json
import re
import sys

FRAME_RE = re.compile(r'^(AS1[1-7])-(\d{2,3})-(\d{4,5})\b', re.M)
STATION_RE = re.compile(r'\bStation\s+(\d+[A-Ca-c]?|LM|ALSEP)\b')
LOC_WORDS = [('ALSEP', r'\bALSEP\b'), ('LM', r'\b(LM|Lunar Module)\b'), ('SEP', r'\bSEP\b')]


def clean(s):
    s = re.sub(r'<[^>]+>', ' ', s)
    return re.sub(r'\s+', ' ', H.unescape(s)).strip()


def parse(path):
    raw = open(path, errors='ignore').read()
    mission = re.search(r'a(1[1-7])/', path) or re.search(r'(1[1-7])', path)
    frames = {}
    # per-frame entries: "AS17-140-21483 (OF300) ( ... )\n<blockquote>caption</blockquote>"
    for m in re.finditer(r'(?m)^(AS1[1-7]-\d{2,3}-\d{4,5})\b[^\n]*\n?\s*<blockquote>(.*?)</blockquote>', raw, re.S):
        fid = m.group(1)
        frames.setdefault(fid, clean(m.group(2)))
    # pans section
    pans = []
    i = raw.find('NAME="Pans"')
    if i < 0:
        i = raw.find('name="Pans"')
    if i >= 0:
        sec = raw[i:]
        # end at next big section header
        j = re.search(r'<font SIZE="\+2"><a NAME="(?!Pans)', sec[10:], re.I)
        if j:
            sec = sec[:j.start() + 10]
        entries = re.split(r'(?=<a [^>]*href="a1[1-7]\.[a-z0-9]+\.html#\d+"[^>]*>\d{3}:\d{2}:\d{2}</a>)', sec)
        for e in entries[1:]:
            t = clean(e)
            title = re.match(r'(\d{3}:\d{2}:\d{2})\s+(.*?)\s*\(', t)
            get, name = (title.group(1), title.group(2)) if title else (None, t[:80])
            fr = []
            for mm in re.finditer(r'(AS1[1-7])-\s*(\d{2,3})-\s*(\d{4,5})\s*(?:to|-|through)\s*(?:(\d{2,3})-)?\s*(\d{2,5})', t):
                pre, mag, a, mag2, b = mm.groups()
                a = int(a)
                b = int(b)
                if b < a:  # abbreviated end like 21509 vs 9, or 20991-027
                    s = str(a)
                    b = int(s[:len(s) - len(str(b))] + str(b))
                if mag2 and mag2 != mag:
                    continue
                if 0 < b - a < 120:
                    fr += [f'{pre}-{mag}-{k}' for k in range(a, b + 1)]
            if fr:
                pans.append({'get': get, 'name': name, 'frames': fr, 'text': t[:600]})
    return frames, pans


def station_of(caption):
    m = STATION_RE.search(caption)
    if m:
        return m.group(1)
    for lab, rx in LOC_WORDS:
        if re.search(rx, caption):
            return lab
    return None


if __name__ == '__main__':
    allp = []
    for p in sys.argv[1:]:
        frames, pans = parse(p)
        allp += pans
        prev = {}
        for fid in sorted(frames, key=lambda f: (f.split('-')[0], int(f.split('-')[1]), int(f.split('-')[2]))):
            cap = frames[fid]
            st = station_of(cap)
            mag = fid.rsplit('-', 1)[0]
            print(json.dumps({'frame': fid, 'station': st, 'caption': cap[:400]}))
    json.dump(allp, open('pans.json', 'w'), indent=1)
    print(f'{len(allp)} pans', file=sys.stderr)
