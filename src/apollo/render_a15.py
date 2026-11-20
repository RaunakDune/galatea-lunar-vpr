"""Render DESPINA constraints for an Apollo 15 station database on the LROC NAC 2 m DTM.

For every traverse station that has pans in pans_final.json, renders views at the station position
(eye height 1.6 m, 49 deg Hasselblad field of view, pitch -5 deg) for 12 headings, lit by the Sun as
seen from the site at the median time of that station's pans (skyfield, DE421 + lunar PA frame).
Writes per view: dem_depth.png, dem_softedge.png, dem_sim.png (shaded DEM, the classical baseline),
skyline.npy and meta.json, plus views.json for the generator.

usage: python render_a15.py --dem NAC_DTM_APOLLO15.TIF --pans pans_final.json --eph eph/ --out views_a15
"""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.expanduser(os.environ.get('DESPINA_DIR', '~/despina')))
from despina.dem import load_dem
from despina.raycast import Camera, RenderConfig, render_view
from despina import constraints as C

# station positions (planetocentric lat, east lon). LM: LROC 2016 coordinates of human features;
# the rest: WUSTL Apollo Analyst's Notebook station layer (traverse navigation tied to the NAC mosaic).
STATIONS = {
    'LM': (26.13239, 3.63330), 'ALSEP': (26.13367, 3.63074),
    '1': (26.03011, 3.60561), '2': (26.01276, 3.58735), '3': (26.07130, 3.61646),
    '4': (26.02916, 3.66131), '6': (25.97581, 3.68056), '6a': (25.97615, 3.67047),
    '7': (25.98308, 3.66752), '9': (26.13098, 3.57170), '9a': (26.13077, 3.56433),
    '10': (26.13705, 3.55867),
}
ALIAS = {'8': 'ALSEP'}           # Dave's Station 8 pan was taken at the ALSEP site
SKIP = {'SEVA'}                  # taken from the LM top hatch, about 7 m up
# Pan times (hours after launch) for pans whose ALSJ entry gives the time only inside its text.
# Station 6 lies outside the valid area of NAC_DTM_APOLLO15 and fails to render.
GET_FALLBACK = {'6': [143.92, 144.24], '6a': [145.12], 'ALSEP': [147.64, 148.04, 164.45], '9': [165.09]}
LAUNCH_UTC = (1971, 7, 26, 13, 34, 0.6)


def get_hours(s):
    h, m, sec = (int(x) for x in s.split(':'))
    return h + m / 60 + sec / 3600


def sun_at(eph_dir, lat, lon, get):
    from skyfield.api import load, load_file
    from skyfield.planetarylib import PlanetaryConstants
    ts = load.timescale()
    eph = load_file(f'{eph_dir}/de421.bsp')
    pc = PlanetaryConstants()
    pc.read_text(load(f'{eph_dir}/moon_080317.tf'))
    pc.read_text(load(f'{eph_dir}/pck00008.tpc'))
    pc.read_binary(load(f'{eph_dir}/moon_pa_de421_1900-2050.bpc'))
    fr = pc.build_frame_named('MOON_ME_DE421')
    site = eph['moon'] + pc.build_latlon_degrees(fr, lat, lon)
    t = ts.tt_jd(ts.utc(*LAUNCH_UTC).tt + get / 24)
    alt, az, _ = site.at(t).observe(eph['sun']).apparent().altaz()
    return float(az.degrees), float(alt.degrees), t.utc_iso()


def lonlat_to_rowcol(dem_path, lat, lon):
    """PDS equirectangular (planetocentric), read from the GeoTIFF's transform and CRS."""
    import rasterio
    from pyproj import CRS, Transformer
    with rasterio.open(dem_path) as ds:
        crs = CRS.from_user_input(ds.crs)
        geo = crs.geodetic_crs
        tr = Transformer.from_crs(geo, crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        row, col = ~ds.transform * (x, y)
        return col, row, ds.shape  # rasterio's inverse gives (col, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dem', required=True)
    ap.add_argument('--pans', required=True)
    ap.add_argument('--eph', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--headings', type=int, default=12)
    ap.add_argument('--pitch', type=float, default=-5.0)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--stations', default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pans = [p for p in json.load(open(a.pans)) if p['mission'] == 15]
    times = {}
    for p in pans:
        s = p['station'].split('-', 1)[1]
        s = ALIAS.get(s, s)
        if s in SKIP or s not in STATIONS:
            continue
        if p['get']:
            times.setdefault(s, []).append(get_hours(p['get']))
    for s, g in GET_FALLBACK.items():
        times.setdefault(s, g)
    dem = load_dem(a.dem, upscale=1.0)
    views = []
    for s, gets in sorted(times.items()):
        if a.stations and s not in a.stations.split(','):
            continue
        lat, lon = STATIONS[s]
        row, col, shape = lonlat_to_rowcol(a.dem, lat, lon)
        sun_az, sun_el, utc = sun_at(a.eph, lat, lon, float(np.median(gets)))
        print(f'station {s}: row {row:.1f} col {col:.1f} of {shape}, sun az {sun_az:.1f} el {sun_el:.1f} ({utc})',
              flush=True)
        for k in range(a.headings):
            hd = 360.0 * k / a.headings
            name = f'st{s}_h{int(hd):03d}'
            d = out / name
            d.mkdir(exist_ok=True)
            cam = Camera(row=row, col=col, heading_deg=hd, pitch_deg=a.pitch, width=a.size, height=a.size)
            res = render_view(dem, cam, RenderConfig(sun_az_deg=sun_az, sun_el_deg=sun_el), device=a.device)
            Image.fromarray(res.inv_depth).save(d / 'dem_depth.png')
            Image.fromarray(res.sim_image).save(d / 'dem_sim.png')
            Image.fromarray(C.soft_edges(res.sim_image, a.device)).save(d / 'dem_softedge.png')
            np.save(d / 'skyline.npy', res.skyline)
            meta = {'station': f'A15-{s}', 'lat': lat, 'lon': lon, 'row': row, 'col': col, 'heading': hd,
                    'pitch': a.pitch, 'sun_az': sun_az, 'sun_el': sun_el, 'utc': utc,
                    'terrain_frac': float(res.terrain_mask.mean()), 'd_far': float(res.d_far.max())}
            json.dump(meta, open(d / 'meta.json', 'w'), indent=1)
            views.append({'name': name, **meta})
    import glob
    views = [json.load(open(m)) | {'name': Path(m).parent.name} for m in sorted(glob.glob(str(out / 'st*' / 'meta.json')))]
    json.dump(views, open(out / 'views.json', 'w'), indent=1)
    print(f'{len(views)} views', flush=True)


if __name__ == '__main__':
    main()
