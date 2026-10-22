"""Recreate the final report table and image directly from four public datasets."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import load_spatial

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
a,s,h,hospitals,ax,ay,polygons,points,_,missing=load_spatial(ROOT,pilot=False)
from shapely.geometry import Point
rows=[]
for i,poly in enumerate(polygons):
    keep=[j for j,p in enumerate(points) if poly.distance(p)<=300]
    rows.append({'FID':a.iloc[i]['사고다발지fid'],'지역명':a.iloc[i]['지점명'],
        '사고건수':int(a.iloc[i]['사고건수']),'사상자수':int(a.iloc[i]['사상자수']),
        '주변정류소수_300m':len(keep),
        '일평균승하차_300m':float(s.iloc[keep].daily_events.sum()),
        '최근접병원직선거리_m':min(Point(ax[i],ay[i]).distance(p) for p in hospitals)})
d=pd.DataFrame(rows)
d.to_csv(OUT/'report_table.csv',index=False,encoding='utf-8-sig')
summary={'areas':len(d),'crash_record_sum':int(d['사고건수'].sum()),'casualty_record_sum':int(d['사상자수'].sum()),
         'areas_with_3_crashes':int((d['사고건수']==3).sum()),'location_stops_excluded_no_common_month_records':missing,
         'hospital_straight_distance_m':{'min':float(d['최근접병원직선거리_m'].min()),'median':float(d['최근접병원직선거리_m'].median()),'max':float(d['최근접병원직선거리_m'].max())}}
(OUT/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
fontpaths=[Path('C:/Windows/Fonts/malgun.ttf'),Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')]
for p in fontpaths:
    if p.exists():
        font_manager.fontManager.addfont(str(p));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(p)).get_name();break
else:plt.rcParams['font.family']=['Noto Sans CJK JP','NanumGothic','DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'svg.fonttype':'none'})
top=d.nlargest(6,'일평균승하차_300m').iloc[::-1]
labels=top['지역명'].str.extract(r'\((.*?) 부근\)')[0]
fig,axes=plt.subplots(1,2,figsize=(12.5,4.9),gridspec_kw={'width_ratios':[1,1.8]})
counts=d['사고건수'].value_counts().sort_index()
bars=axes[0].bar(counts.index.astype(str),counts.values,color='#4978AD',width=.6)
axes[0].bar_label(bars,padding=4)
axes[0].set(xlabel='사고 다발지역별 사고건수 (건)',ylabel='해당 지역 수 (곳)',title='① 24곳 중 17곳의 사고건수가 3건',ylim=(0,20))
bars=axes[1].barh(labels,top['일평균승하차_300m']/10000,color='#178B84',height=.62)
axes[1].bar_label(bars,labels=[f'{v:,.0f}건 / 사고 {int(c)}건' for v,c in zip(top['일평균승하차_300m'],top['사고건수'])],padding=5,fontsize=10)
axes[1].set(xlabel='주변 정류소 일평균 승하차 (만 건)',title='② 주변 대중교통 이용 규모 상위 6곳',xlim=(0,9.7))
for i,axp in enumerate(axes):
    axp.spines[['top','right']].set_visible(False);axp.grid(axis='x' if i else 'y',alpha=.16);axp.set_axisbelow(True)
fig.suptitle('서울 결빙 사고 다발지역: 사고 이력과 주변 버스 이용 규모',fontweight='bold',fontsize=16,y=.98)
fig.text(.02,.045,'사고: 2021~2025년 집계자료 | 승하차: 2026년 1·2월 월별 일평균의 평균, 양월 공통 정류소',fontsize=9,color='#444444')
fig.text(.02,.008,'범위: 사고지역 폴리곤 경계에서 300m 이내. 승하차는 이용 건수이며 영향 인구·운행 지연·사고 감소 효과를 뜻하지 않음.',fontsize=9,color='#444444')
fig.tight_layout(rect=(0,.10,1,.92))
fig.savefig(OUT/'report_figure.png',dpi=220,bbox_inches='tight')
fig.savefig(OUT/'report_figure.pdf',bbox_inches='tight')
fig.savefig(OUT/'report_figure.svg',bbox_inches='tight')
plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2))
