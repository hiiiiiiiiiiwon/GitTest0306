"""Audit packaged inputs and report calculations; no policy-effect estimation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import load_spatial

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'

def require(ok, message):
    if not ok:
        raise ValueError(message)

manifest = json.loads((ROOT/'data/checksums.json').read_text(encoding='utf-8'))
for name, expected in manifest.items():
    require(hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest() == expected,
            f'Input checksum mismatch: {name}')
a,s,h,hp,ax,ay,polys,points,_,_ = load_spatial(ROOT, pilot=False)
b = pd.read_csv(ROOT/'data/bus_daily.csv.gz', dtype={'표준버스정류장ID':str,'month':str})
require(not b.duplicated(['표준버스정류장ID','month']).any(), 'Duplicate monthly stop IDs')
require(np.isfinite(b.daily).all() and (b.daily >= 0).all(), 'Invalid ridership')
require(all(np.isfinite([p.x,p.y]).all() for p in points+hp), 'Invalid coordinates')
require(pd.to_numeric(a['사고건수']).sum()==78, 'Crash total differs from report')
require(pd.to_numeric(a['사상자수']).sum()==116, 'Casualty total differs from report')
monthly = b.pivot(index='표준버스정류장ID',columns='month',values='daily')
rows=[]
for i,poly in enumerate(polys):
    row={'FID':str(a.iloc[i]['사고다발지fid']),'지역명':a.iloc[i]['지점명']}
    for radius in (100,300,500):
        mask=np.array([poly.distance(p)<=radius for p in points])
        ids=s.loc[mask,'NODE_ID']
        row[f'bus_{radius}m']=float(s.loc[mask,'daily_events'].sum())
        if radius==300:
            jan=float(monthly.loc[ids,'202601'].sum())
            feb=float(monthly.loc[ids,'202602'].sum())
            row.update(jan_daily=jan,feb_daily=feb,day_weighted_daily=(31*jan+28*feb)/59)
    require(row['bus_100m']<=row['bus_300m']<=row['bus_500m'], 'Nonmonotonic spatial sum')
    rows.append(row)
d=pd.DataFrame(rows)
report=pd.read_csv(OUT/'report_table.csv',dtype={'FID':str}).set_index('FID')
require(set(d.FID)==set(report.index), 'Report area IDs mismatch')
require(len(d)==24 and report.index.is_unique, 'Expected 24 unique report areas')
require(np.allclose(d.bus_300m,report.loc[d.FID,'일평균승하차_300m'],rtol=0,atol=1e-6), 'Report ridership mismatch')
for field, source in [('사고건수','사고건수'),('사상자수','사상자수')]:
    require(np.array_equal(report.loc[d.FID,field],pd.to_numeric(a[source])), f'Report mismatch: {field}')
from shapely.geometry import Point
distances=[min(Point(x,y).distance(p) for p in hp) for x,y in zip(ax,ay)]
require(np.allclose(distances,report.loc[d.FID,'최근접병원직선거리_m'],atol=1e-6,rtol=0), 'Hospital distances mismatch')
summary=json.loads((OUT/'report_summary.json').read_text(encoding='utf-8'))
require((summary['areas'],summary['crash_record_sum'],summary['casualty_record_sum'],summary['areas_with_3_crashes'])==(24,78,116,17), 'Report summary mismatch')
checks=[]
for col in ('bus_100m','bus_500m','jan_daily','feb_daily','day_weighted_daily'):
    # Spearman handles ties. Top-six is descriptive, NOT a validated management rank.
    checks.append({'comparison':col,'spearman_vs_report_300m':float(spearmanr(d.bus_300m,d[col]).statistic),
        'top6_overlap':len(set(d.nlargest(6,'bus_300m').FID)&set(d.nlargest(6,col).FID))})
d.to_csv(OUT/'report_sensitivity_values.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(checks).to_csv(OUT/'report_sensitivity_summary.csv',index=False,encoding='utf-8-sig')
result={'status':'PASS','verified_input_files':len(manifest),'areas':len(d),
    'checks':['input SHA-256','unique stop-month keys','finite nonnegative ridership','finite coordinates',
              '24 area IDs','78 crash records','116 casualty records','report spatial sums','hospital distances','nested distance totals'],
    'sensitivity':checks,
    'scope':'Spatial sums include polygon interior and points within the specified exterior distance. Common-month stops only. Correlations describe ridership rankings, not management priorities or causal effects.',
    'carbon':'No measured fuel/electricity data; no emissions reduction estimate.'}
(OUT/'submission_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
