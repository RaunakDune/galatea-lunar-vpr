"""Unpack the LunarLoc traverses (github.com/mit-acl/lunarloc-data, release v1.0.0) for retrieval.

For each traverse_<k>.lac (a tar.gz archive) this extracts the FrontLeft grayscale images (1280 x 720,
70 deg horizontal field of view, one every 40 simulator frames) and writes <out>/traverse_<k>/meta.json
with, per image, the camera pose and the boulders LunarLoc detected in that frame:
  camera position = rover position + R(roll, pitch, yaw) @ camera offset (both from the archive),
  camera heading  = rover yaw + camera yaw (deg, counter-clockwise from +x),
  boulders        = the 3-D boulder centroids of custom/boulders.csv for the same simulator frame
                    (LunarLoc's own FastSAM + stereo front end; world coordinates), or [] if none.
usage: python lunarloc_prep.py --lac_dir ~/lunarloc/v1 --out ~/lunarloc/ext
"""
import os, io, re, csv, ast, glob, json, tarfile, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--lac_dir', default=os.path.expanduser('~/lunarloc/v1'))
ap.add_argument('--out', default=os.path.expanduser('~/lunarloc/ext'))
args = ap.parse_args()


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rows(tf, name):
    m = next(x for x in tf.getmembers() if x.name.lstrip('./').endswith(name))
    return list(csv.DictReader(io.TextIOWrapper(tf.extractfile(m), 'utf-8')))


for f in sorted(glob.glob(f'{args.lac_dir}/traverse_*.lac'), key=lambda s: int(re.findall(r'\d+', os.path.basename(s))[0])):
    k = int(re.findall(r'\d+', os.path.basename(f))[0])
    d = f'{args.out}/traverse_{k}'
    if os.path.exists(f'{d}/meta.json'):
        continue
    os.makedirs(f'{d}/img', exist_ok=True)
    with tarfile.open(f) as tf:
        fr = {float(r['frame']): r for r in rows(tf, 'frames.csv') if r['frame']}
        cam = rows(tf, 'FrontLeft_frames.csv')
        try:
            bl = {float(r['frame']): ast.literal_eval(r['boulders']) for r in rows(tf, 'boulders.csv')}
        except StopIteration:
            bl = {}
        members = {os.path.basename(m.name): m for m in tf.getmembers() if '/FrontLeft/grayscale/' in m.name}
        recs = []
        for c in cam:
            if c.get('enable', 'True') != 'True' or not c['grayscale'] or c['grayscale'] not in members:
                continue
            t = float(c['frame']); r = fr[t]
            R = rot(float(r['roll']), float(r['pitch']), float(r['yaw']))
            p = np.array([float(r['x']), float(r['y']), float(r['z'])]) + R @ np.array(
                [float(c['camera_x']), float(c['camera_y']), float(c['camera_z'])])
            with open(f'{d}/img/{c["grayscale"]}', 'wb') as o:
                o.write(tf.extractfile(members[c['grayscale']]).read())
            recs.append({'file': c['grayscale'], 'frame': t, 'x': p[0], 'y': p[1], 'z': p[2],
                         'yaw': float(np.degrees(float(r['yaw']) + float(c['camera_yaw']))) % 360,
                         'boulders': [list(map(float, b)) for b in bl.get(t, [])]})
    json.dump({'traverse': k, 'images': recs}, open(f'{d}/meta.json', 'w'))
    nb = np.mean([len(r['boulders']) for r in recs])
    print(f'traverse {k}: {len(recs)} images, mean {nb:.1f} boulders per image', flush=True)
