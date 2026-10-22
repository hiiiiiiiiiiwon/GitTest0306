"""Shared spatial preparation. Distances are metres in EPSG:5179."""
import json
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

def load_spatial(root, pilot=False):
    data=root/'data'
    def read(name):return pd.read_csv(data/name,dtype=str,keep_default_na=False)
    a=read('accidents.csv.gz');s=read('stops.csv.gz');h=read('hospitals.csv.gz')
    assert len(a)==24 and a['사고다발지fid'].is_unique
    assert s.NODE_ID.is_unique and len(h)==283
    s=s[s['정류소타입']!='한강선착장'].copy()
    b=read('bus_daily.csv.gz');b['daily']=pd.to_numeric(b['daily'])
    monthly=b.pivot(index='표준버스정류장ID',columns='month',values='daily')[['202601','202602']]
    # Pilot: an absent month is zero. Report: only IDs observed in both months.
    combined=monthly.fillna(0).mean(axis=1) if pilot else monthly.dropna().mean(axis=1)
    s['daily_events']=s.NODE_ID.map(combined)
    s=s[s.daily_events.notna()].copy()
    tr=Transformer.from_crs(4326,5179,always_xy=True)
    sx,sy=tr.transform(pd.to_numeric(s['X좌표']).to_numpy(),pd.to_numeric(s['Y좌표']).to_numpy())
    points=[Point(x,y) for x,y in zip(sx,sy)]
    ht=Transformer.from_crs(5174,5179,always_xy=True)
    hx,hy=ht.transform(pd.to_numeric(h['좌표정보(X)']).to_numpy(),pd.to_numeric(h['좌표정보(Y)']).to_numpy())
    hospitals=[Point(x,y) for x,y in zip(hx,hy)]
    ax,ay=tr.transform(pd.to_numeric(a['경도']).to_numpy(),pd.to_numeric(a['위도']).to_numpy())
    polygons=[transform(tr.transform,shape(json.loads(v))) for v in a['다발지역폴리곤']]
    assert all(p.is_valid and not p.is_empty for p in polygons)
    provenance=json.loads((data/'provenance.json').read_text(encoding='utf-8'))
    coverage=provenance['bus_join_coverage']
    # Source-wide missing counts were computed before the 500m package filter.
    key='original_pilot_missing_location_stops' if pilot else 'original_report_missing_location_stops'
    missing_stop=provenance['packaging'][key]
    return a,s,h,hospitals,ax,ay,polygons,points,coverage,missing_stop
