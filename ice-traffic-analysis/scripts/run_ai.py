"""Reproducible exploratory AI validation; no drone detection or routing model."""
from pathlib import Path
import sys, json, platform
ROOT=Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import sklearn, shapely, pyproj, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from shapely.geometry import shape, Point
from shapely.ops import transform
from pyproj import Transformer
from threadpoolctl import threadpool_limits

OUT=ROOT/'results'/'ai'
OUT.mkdir(exist_ok=True)
from common import load_spatial
a,s,h,hospital_points,ax,ay,polys,stop_points,coverage,missing_stop = load_spatial(ROOT, pilot=True)
counts=pd.to_numeric(a['사고건수']).to_numpy(float)
cas=pd.to_numeric(a['사상자수']).to_numpy(float)
assert (counts>0).all()
base=pd.DataFrame({'FID':a['사고다발지fid'],'지역명':a['지점명'],'사고건수':counts,'사고당사상자수':cas/counts})
base['최근접병원직선거리_m']=[min(Point(x,y).distance(p) for p in hospital_points) for x,y in zip(ax,ay)]
for r in [100,300,500]:
    arrays=[np.array([p.distance(st)<=r for st in stop_points]) for p in polys]
    base[f'주변정류소수_{r}m']=[int(m.sum()) for m in arrays]
    base[f'일평균승하차건수_{r}m']=[float(s.loc[m,'daily_events'].sum()) for m in arrays]

FEATURES=['사고건수','사고당사상자수','일평균승하차건수_300m','최근접병원직선거리_m']
def feature_matrix(radius=300):
    d=base[['사고건수','사고당사상자수',f'일평균승하차건수_{radius}m','최근접병원직선거리_m']].to_numpy(float)
    return np.log1p(d)
scaler=StandardScaler()
X=scaler.fit_transform(feature_matrix())
assert np.isfinite(X).all()
def fit(x,k,seed=42,n_init=20):return KMeans(n_clusters=k,random_state=seed,n_init=n_init).fit(x)
def score(x,model):
    lab=model.labels_; sizes=np.bincount(lab,minlength=model.n_clusters)
    return {'k':model.n_clusters,'silhouette':float(silhouette_score(x,lab)),'sizes':sizes.tolist(),'min_cluster_size':int(sizes.min())}

with threadpool_limits(limits=1):
    models=[fit(X,k) for k in [2,3,4]]
    candidates=[score(X,m) for m in models]
    eligible=[m for m in models if np.bincount(m.labels_,minlength=m.n_clusters).min()>=3]
    chosen=max(eligible or models,key=lambda m:silhouette_score(X,m.labels_))
    lab=chosen.labels_; k=chosen.n_clusters
    seed_ari=[adjusted_rand_score(lab,fit(X,k,seed=i,n_init=10).labels_) for i in range(50)]
    loo=[]
    for i in range(len(X)):
        mask=np.arange(len(X))!=i
        m=fit(X[mask],k,n_init=20)
        loo.append(float(adjusted_rand_score(lab[mask],m.labels_)))
    radii=[]
    for r in [100,300,500]:
        xr=StandardScaler().fit_transform(feature_matrix(r))
        mr=fit(xr,k)
        radii.append({'radius_m':r,'ARI_vs_300m':float(adjusted_rand_score(lab,mr.labels_)),'silhouette':float(silhouette_score(xr,mr.labels_))})
    rng=np.random.default_rng(42)
    null=[]
    # Same k search and minimum-size rule for each independently permuted feature matrix.
    for _ in range(100):
        xp=np.column_stack([rng.permutation(X[:,j]) for j in range(X.shape[1])])
        ms=[fit(xp,j,n_init=10) for j in [2,3,4]]
        es=[m for m in ms if np.bincount(m.labels_,minlength=m.n_clusters).min()>=3]
        null.append(max(float(silhouette_score(xp,m.labels_)) for m in (es or ms)))
    ablation=[]
    for j,col in enumerate(FEATURES):
        m=fit(np.delete(X,j,axis=1),k)
        ablation.append({'removed_feature':col,'ARI_vs_full':float(adjusted_rand_score(lab,m.labels_))})

base['군집번호']=lab+1
base.to_csv(OUT/'AI_입력지표_군집결과.csv',index=False,encoding='utf-8-sig')
profile=base.groupby('군집번호')[FEATURES].agg(['count','mean','median'])
profile.to_csv(OUT/'군집별_지표요약.csv',encoding='utf-8-sig')
pd.DataFrame(candidates).to_csv(OUT/'군집수_비교.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(radii).to_csv(OUT/'거리기준_민감도.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(ablation).to_csv(OUT/'변수제외_민감도.csv',index=False,encoding='utf-8-sig')
base[FEATURES].corr(method='spearman').to_csv(OUT/'입력변수_상관.csv',encoding='utf-8-sig')
observed=float(silhouette_score(X,lab))
result={'scope':'서울 사고지역 24개에 대한 탐색적 관리 특성 군집분석. 드론 탐지/결빙예측/도로망 접근성 분석 아님.',
 'features':FEATURES,'log1p_then_standard_scaler':True,'k_search':candidates,'selected_k':k,'selected_silhouette':observed,
 'seed_ARI_min':float(min(seed_ari)),'seed_ARI_mean':float(np.mean(seed_ari)),
 'leave_one_out_ARI_min':float(min(loo)),'leave_one_out_ARI_median':float(np.median(loo)),
 'radius_sensitivity':radii,'feature_ablation':ablation,'permuted_best_silhouette_mean':float(np.mean(null)),
 'permutation_exceedance_fraction':float((1+sum(v>=observed for v in null))/101),
 'null_note':'탐색적 참고값이며 표본 설계에 기반한 유의성 검정 또는 예측 정확도가 아님.',
 'bus_join_coverage':coverage,'location_stops_without_matched_usage':missing_stop,'hospital_points':len(h),
 'hospital_note':'영업중·좌표보유 병원/종합병원 현황. 최근접 직선거리는 도로 접근시간이나 통행장애 영향이 아님.',
 'temporal_note':'2021~2025 사고 집계, 2026년1~2월 이용량, 확보시점 병원현황 결합. 인과효과 추정 아님.',
 'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'pyproj':pyproj.__version__,'shapely':shapely.__version__}}
(OUT/'검증결과.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
joblib.dump({'model':chosen,'scaler':scaler,'feature_order':FEATURES,'transformation':'log1p','scope':result['scope']},OUT/'관리특성_군집모형.joblib')
print(json.dumps(result,ensure_ascii=False,indent=2))
