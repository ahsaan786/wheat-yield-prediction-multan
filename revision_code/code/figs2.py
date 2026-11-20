import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json
from matplotlib.lines import Line2D
from pipe import base_df, rmse
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10.5,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','legend.frameon':False})
O='figs/'
DIST=['Khanewal','Lodhran','Multan','Vehari']
DCOL={'Khanewal':'#1f77b4','Lodhran':'#2ca02c','Multan':'#d62728','Vehari':'#9467bd'}; DMK={'Khanewal':'o','Lodhran':'s','Multan':'^','Vehari':'D'}
M=['XGB','LSTM','TCN','XGB+LSTM','TCN+XGB','LSTM+TCN','XGB+LSTM+TCN']
LAB={'XGB':'XGBoost'}
MCOL={'XGB':'#ff7f0e','LSTM':'#1f77b4','TCN':'#2ca02c','XGB+LSTM':'#8c564b','TCN+XGB':'#e377c2','LSTM+TCN':'#4d4d4d','XGB+LSTM+TCN':'#17becf'}
TR='Trend only (Theil–Sen)'
P=pd.read_csv('final_preds.csv').set_index(['District','Year'])
def met(c): e=P[c]-P.obs; return dict(R2=1-(e**2).sum()/((P.obs-P.obs.mean())**2).sum(),RMSE=rmse(e),MAE=e.abs().mean())

def fig3():
    df=base_df(); d=df[df.Year<=2020]
    cols=['NDVI_mean','LST_mean_C','Rain_log','VPD_kPa','HeatStress_log','ColdStress_log','y']
    nm=['NDVI (wheat mask)','LST','Rain (log)','VPD','Heat stress (log)','Cold stress (log)','Yield']
    C=d[cols].corr().values
    fig,ax=plt.subplots(figsize=(7.2,6.2),dpi=300)
    im=ax.imshow(C,cmap='RdBu_r',vmin=-1,vmax=1)
    for i in range(7):
        for j in range(7): ax.text(j,i,f'{C[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(C[i,j])>0.6 else 'black')
    ax.set_xticks(range(7)); ax.set_xticklabels(nm,rotation=40,ha='right'); ax.set_yticks(range(7)); ax.set_yticklabels(nm)
    ax.spines[:].set_visible(False); fig.colorbar(im,fraction=0.046,pad=0.03).set_label('Pearson r')
    ax.set_title('Pre-test period 2001–2020 (n = 80)')
    fig.savefig(O+'fig3_corr.png',bbox_inches='tight',facecolor='white'); return C

def fig4():
    m={c:met(c) for c in M+[TR]}
    fig=plt.figure(figsize=(13,5.2),dpi=300); gs=fig.add_gridspec(1,3,width_ratios=[1,1.15,1.25],wspace=0.32)
    a=fig.add_subplot(gs[0]); b=fig.add_subplot(gs[1]); c=fig.add_subplot(gs[2]); x=np.arange(len(M)); lab=[LAB.get(s,s) for s in M]
    r2=[m[n]['R2'] for n in M]; a.bar(x,r2,color=[MCOL[n] for n in M],edgecolor='white',width=0.72)
    for i,v in enumerate(r2): a.text(i,v+0.012,f'{v:.2f}',ha='center',fontsize=8.5,fontweight='bold')
    a.axhline(m[TR]['R2'],ls=':',c='black',lw=1); a.text(1.0,m[TR]['R2']+0.012,'trend only',ha='center',fontsize=8)
    a.axvline(2.5,ls='--',c='grey',lw=0.8); a.set_xticks(x); a.set_xticklabels(lab,rotation=45,ha='right',fontsize=8.8)
    a.set_ylim(0,0.75); a.set_ylabel('R²'); a.set_title('(a) Coefficient of determination')
    a.text(1,0.71,'standalone',ha='center',fontsize=8.5,color='grey'); a.text(4.5,0.71,'SERWI hybrids',ha='center',fontsize=8.5,color='grey')
    w=0.38; rm=[m[n]['RMSE'] for n in M]; ma=[m[n]['MAE'] for n in M]
    b.bar(x-w/2,rm,w,color='#3a6ea5',label='RMSE'); b.bar(x+w/2,ma,w,color='#f0a58a',label='MAE')
    for i in range(len(M)): b.text(i-w/2,rm[i]+3,f'{rm[i]:.0f}',ha='center',fontsize=7.6); b.text(i+w/2,ma[i]+3,f'{ma[i]:.0f}',ha='center',fontsize=7.6)
    b.axvline(2.5,ls='--',c='grey',lw=0.8); b.axhline(m[TR]['RMSE'],ls=':',c='black',lw=1); b.text(4.0,m[TR]['RMSE']+4,'trend-only RMSE',ha='center',fontsize=8)
    b.set_xticks(x); b.set_xticklabels(lab,rotation=45,ha='right',fontsize=8.8); b.set_ylim(0,240); b.set_ylabel('Error (kg/ha)'); b.set_title('(b) RMSE and MAE'); b.legend(loc='upper left',ncol=2,fontsize=8.5)
    rows=M+[TR]; mat=np.array([[rmse(P.loc[dd][n]-P.loc[dd].obs) for dd in DIST] for n in rows])
    im=c.imshow(mat,cmap='YlOrRd',aspect='auto',vmin=80,vmax=300)
    for i in range(mat.shape[0]):
        for j in range(4): c.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',fontsize=8.6,color='white' if mat[i,j]>240 else 'black',fontweight='bold' if rows[i]=='XGB' else 'normal')
    c.set_xticks(range(4)); c.set_xticklabels(DIST,fontsize=9); c.set_yticks(range(len(rows))); c.set_yticklabels([LAB.get(s,'Trend only' if s==TR else s) for s in rows],fontsize=9)
    c.spines[:].set_visible(False); c.set_title('(c) Test RMSE by district (kg/ha)'); c.axhline(6.5,c='white',lw=3)
    fig.colorbar(im,ax=c,fraction=0.046,pad=0.03).set_label('RMSE (kg/ha)',fontsize=9)
    fig.savefig(O+'fig4_performance.png',bbox_inches='tight',facecolor='white')

def fig5():
    fig=plt.figure(figsize=(13,6.6),dpi=300); gs=fig.add_gridspec(2,3,width_ratios=[1.35,1,1],wspace=0.28,hspace=0.42)
    a=fig.add_subplot(gs[:,0]); lo,hi=3250,4250; xx=np.linspace(lo,hi,50)
    a.fill_between(xx,xx*0.95,xx*1.05,color='#d9d9d9',alpha=0.6,lw=0,label='±5% of observed'); a.plot([lo,hi],[lo,hi],'k--',lw=1.2,label='1:1 line')
    for dd in DIST:
        s=P.loc[dd]; a.scatter(s.obs,s.XGB,s=70,marker=DMK[dd],color=DCOL[dd],edgecolor='black',lw=0.5,zorder=3,label=dd)
    mm=met('XGB'); mt=met(TR); within=(np.abs(P.XGB-P.obs)<=0.05*P.obs).mean()*100
    a.text(0.04,0.96,f'XGBoost (operational)\nn = 20 (4 districts × 5 years)\nR² = {mm["R2"]:.2f}\nRMSE = {mm["RMSE"]:.1f} kg/ha\nMAE = {mm["MAE"]:.1f} kg/ha\n{within:.0f}% of points within ±5%\nTrend-only RMSE = {mt["RMSE"]:.1f} kg/ha',
           transform=a.transAxes,va='top',fontsize=9,bbox=dict(boxstyle='round',fc='white',ec='#bbbbbb'))
    a.set_xlim(lo,hi); a.set_ylim(lo,hi); a.set_aspect('equal'); a.set_xlabel('Observed yield (kg/ha)'); a.set_ylabel('Predicted yield (kg/ha)')
    a.set_title('(a) All districts, 2021–2025 test period'); a.legend(loc='lower right',fontsize=8.5)
    for k,dd in enumerate(DIST):
        ax=fig.add_subplot(gs[k//2,1+k%2]); s=P.loc[dd]; yr=s.index.values
        ax.plot(yr,s.obs,'-o',c='black',lw=2,ms=5,zorder=4); ax.plot(yr,s.XGB,'-',marker=DMK[dd],c=DCOL[dd],lw=2,ms=5,zorder=3)
        ax.plot(yr,s['TCN+XGB'],'--',c='#555555',lw=1.3); ax.plot(yr,s[TR],':',c='#9a9a9a',lw=1.8)
        ax.set_title(f'({"bcde"[k]}) {dd}',color=DCOL[dd])
        ax.text(0.03,0.95,f'RMSE: XGBoost {rmse(s.XGB-s.obs):.0f}  |  trend only {rmse(s[TR]-s.obs):.0f}',transform=ax.transAxes,fontsize=8,va='top')
        ax.set_xticks(range(2021,2026)); ax.set_ylim(3150,4300); ax.grid(alpha=0.25)
        if k%2==0: ax.set_ylabel('Yield (kg/ha)')
    hdl=[Line2D([],[],c='black',marker='o',lw=2),Line2D([],[],c='#333333',marker='s',lw=2,mfc='white'),Line2D([],[],c='#555555',ls='--',lw=1.3),Line2D([],[],c='#9a9a9a',ls=':',lw=1.8)]
    fig.legend(hdl,['Observed','XGBoost (operational; district colour)','TCN+XGB (best SERWI hybrid)','Trend only (no climate input)'],loc='lower center',ncol=2,bbox_to_anchor=(0.66,-0.06),fontsize=9)
    fig.savefig(O+'fig5_obs_pred.png',bbox_inches='tight',facecolor='white')

def fig6():
    fig,axs=plt.subplots(2,2,figsize=(12,7.2),dpi=300,sharex=True,sharey=True)
    for k,dd in enumerate(DIST):
        ax=axs[k//2,k%2]; s=P.loc[dd]
        for n in M+[TR]:
            ax.plot(s.index,s.obs-s[n],marker='o',ms=3.5,lw=2.4 if n=='XGB' else 1.2,c=MCOL.get(n,'#9a9a9a'),ls=':' if n==TR else '-',
                    label=(LAB.get(n,n) if n!=TR else 'Trend only')+(' (operational)' if n=='XGB' else ''),zorder=4 if n=='XGB' else 2)
        ax.axhline(0,c='black',lw=0.8,ls=':'); ax.set_title(f'({"abcd"[k]}) {dd}',color=DCOL[dd]); ax.grid(alpha=0.25)
        if k%2==0: ax.set_ylabel('Residual, observed − predicted (kg/ha)')
        ax.set_xticks(range(2021,2026))
    h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(0.5,-0.04),fontsize=9)
    fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(O+'fig6_residuals.png',bbox_inches='tight',facecolor='white')

def fig7(RT):
    S=pd.read_csv('seeds.csv')
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5),dpi=300,gridspec_kw={'width_ratios':[1.2,1]})
    ms=['XGB','LSTM','TCN','TCN+XGB','XGB+LSTM',TR]; yrs=range(2021,2026); w=0.13
    for i,n in enumerate(ms):
        v=[rmse(RT.loc[RT.Year==y,n]-RT.loc[RT.Year==y,'obs']) for y in yrs]
        a.bar(np.arange(5)+(i-2.5)*w,v,w,color=MCOL.get(n,'#bdbdbd'),label=(LAB.get(n,n) if n!=TR else 'Trend only')+f' ({rmse(RT[n]-RT.obs):.0f})')
    a.set_xticks(range(5)); a.set_xticklabels(yrs); a.set_ylabel('RMSE across districts (kg/ha)'); a.set_title('(a) Rolling-origin one-year-ahead forecasts')
    a.legend(fontsize=8.3,ncol=3,title='model (overall RMSE)',title_fontsize=8.3,loc='upper center'); a.set_ylim(0,370); a.grid(axis='y',alpha=0.25)
    bm=['LSTM','TCN','XGB+LSTM','TCN+XGB','LSTM+TCN','XGB+LSTM+TCN']
    data=[S[n].clip(lower=-3) for n in bm]
    bp=b.boxplot(data,patch_artist=True,widths=0.55,showfliers=True)
    for p_,n in zip(bp['boxes'],bm): p_.set_facecolor(MCOL[n]); p_.set_alpha(0.6)
    xg=json.load(open('xgb_seeds.json')); b.axhspan(xg['min'],xg['max'],color=MCOL['XGB'],alpha=0.2); b.axhline(xg['mean'],c=MCOL['XGB'],lw=2)
    b.text(0.55,xg['mean']+0.07,'XGBoost, 20 seeds',ha='left',fontsize=8.3,color='#b35900')
    b.axhline(0,c='black',lw=0.8,ls=':'); b.set_xticks(range(1,7)); b.set_xticklabels([LAB.get(n,n) for n in bm],rotation=30,ha='right')
    b.set_ylim(-3.15,0.95); b.set_ylabel('Test R² of a single seed'); b.set_title('(b) Seed-to-seed spread, 20 seeds (values < −3 clipped)')
    fig.tight_layout(); fig.savefig(O+'fig7_rolling_seeds.png',bbox_inches='tight',facecolor='white')

def fig8():
    D_=pd.read_csv('dm.csv'); D_=D_[D_.model!='Climatology'].sort_values('DM')
    fig,ax=plt.subplots(figsize=(8.5,5.6),dpi=300)
    col=['#3a6ea5' if q<0.10 else '#9ecae1' for q in D_.q]
    ax.barh(range(len(D_)),D_.DM,color=col)
    for i,(s,p,q) in enumerate(zip(D_.DM,D_.p,D_.q)): ax.text(s-0.05,i,f'p = {p:.3f}, q = {q:.2f}',ha='right',va='center',fontsize=8)
    ax.set_yticks(range(len(D_))); ax.set_yticklabels([LAB.get(m,m) for m in D_.model],fontsize=9); ax.axvline(0,c='black',lw=0.8)
    ax.set_xlim(-4.6,0.3); ax.set_xlabel('HLN-corrected DM statistic (negative = XGBoost more accurate)')
    ax.set_title('Diebold–Mariano tests: XGBoost against each model and baseline')
    fig.savefig(O+'fig8_dm.png',bbox_inches='tight',facecolor='white')

def fig9():
    S=pd.read_csv('shap_xgb_ts_prev20.csv'); X=pd.read_csv('shap_X_ts_prev20.csv')
    nm={'NDVI_mean':'NDVI (Mar–Apr, wheat mask)','LST_mean_C':'LST (Nov–Apr)','Rain_log':'Rainfall (log)','VPD_kPa':'VPD (Mar–Apr)','HeatStress_log':'Heat-stress days (log)',
        'ColdStress_log':'Cold-stress days (log)','Yield_lag1':'Yield lag (detrended)','NDVI_x_VPD':'NDVI × VPD','LST_x_Heat':'LST × heat stress','dcode':'District identity'}
    order=S.abs().mean().sort_values().index
    fig=plt.figure(figsize=(13.5,6.4),dpi=300); gs=fig.add_gridspec(2,5,width_ratios=[1.6,0.04,0.12,1,1],wspace=0.42,hspace=0.45)
    a=fig.add_subplot(gs[:,0]); rng=np.random.default_rng(0)
    for i,c in enumerate(order):
        v=X[c]; z=(v-v.min())/(v.max()-v.min()+1e-9)
        a.scatter(S[c],i+rng.uniform(-0.28,0.28,len(v)),c=z,cmap='coolwarm',s=11,lw=0)
    a.set_yticks(range(len(order))); a.set_yticklabels([nm[c]+f'  ({S[c].abs().mean():.0f})' for c in order],fontsize=9)
    a.axvline(0,c='grey',lw=0.8); a.set_xlabel('SHAP value (kg/ha; effect on predicted yield residual)')
    a.set_title('(a) Signed SHAP values, XGBoost (mean |SHAP| in brackets)')
    sm=plt.cm.ScalarMappable(cmap='coolwarm'); cb=fig.colorbar(sm,cax=fig.add_subplot(gs[:,1])); cb.set_ticks([0,1]); cb.set_ticklabels(['low','high']); cb.set_label('feature value')
    for k,c in enumerate(['ColdStress_log','LST_mean_C','NDVI_mean','Rain_log']):
        ax=fig.add_subplot(gs[k//2,3+k%2]); ax.scatter(X[c],S[c],s=12,c='#3a6ea5',alpha=0.7,lw=0)
        o=np.argsort(X[c].values); ax.plot(X[c].values[o],pd.Series(S[c].values[o]).rolling(15,center=True,min_periods=5).mean(),c='#d62728',lw=2)
        ax.axhline(0,c='grey',lw=0.8); ax.set_xlabel(nm[c],fontsize=9); ax.set_ylabel('SHAP (kg/ha)',fontsize=9); ax.set_title(f'({"bcde"[k]})',loc='left')
    fig.savefig(O+'fig9_shap.png',bbox_inches='tight',facecolor='white')

def fig10():
    R=pd.read_csv('proj_xgb_ts_prev20.csv'); df=base_df()
    fig,axs=plt.subplots(2,2,figsize=(12.5,7.6),dpi=300,sharex=True)
    for k,dd in enumerate(DIST):
        ax=axs[k//2,k%2]; hh=df[df.District==dd].sort_values('Year')
        ax.plot(hh.Year,hh.y,'-o',c='black',ms=3,lw=1.4,label='Observed 2001–2025')
        g=lambda s,t: R[(R.District==dd)&(R.Scenario==s)&(R.trend==t)].sort_values('Year')
        Af,Ac,Bf,Bc=g('A','frozen'),g('A','continued'),g('B','frozen'),g('B','continued')
        ax.fill_between(Af.Year,np.minimum.reduce([Af.pred.values,Ac.pred.values,Bf.pred.values,Bc.pred.values]),np.maximum.reduce([Af.pred.values,Ac.pred.values,Bf.pred.values,Bc.pred.values]),color=DCOL[dd],alpha=0.15,lw=0,label='Range of the four scenario–trend runs')
        ax.plot(Af.Year,Af.pred,'-',marker='s',ms=4,c=DCOL[dd],lw=2.2,label='Scenario A, trend frozen at 2025')
        ax.plot(Bf.Year,Bf.pred,'--',marker='^',ms=4,c=DCOL[dd],lw=1.6,label='Scenario B, trend frozen at 2025')
        ax.plot(Ac.Year,Ac.pred,':',c=DCOL[dd],lw=1.6,label='Scenario A, Theil–Sen trend continued')
        rec=hh[hh.Year>=2021].y.mean(); ax.hlines(rec,2021,2030,colors='#666666',linestyles=':',lw=1.2,label='Observed 2021–2025 mean')
        ax.axvline(2025.5,c='grey',lw=0.8,ls='--'); ax.text(2025.6,2520,'projection',fontsize=8,color='grey')
        ax.text(0.02,0.95,f'2026–2030 mean, frozen trend: A {Af.pred.mean():,.0f} | B {Bf.pred.mean():,.0f} kg/ha',transform=ax.transAxes,fontsize=8.3,va='top')
        ax.set_title(f'({"abcd"[k]}) {dd}',color=DCOL[dd]); ax.set_ylim(2450,4500); ax.grid(alpha=0.25)
        if k%2==0: ax.set_ylabel('Wheat yield (kg/ha)')
        if k>=2: ax.set_xlabel('Year')
    h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.06),fontsize=9)
    fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(O+'fig10_projections.png',bbox_inches='tight',facecolor='white')

if __name__=='__main__':
    import sys
    for f in sys.argv[1:]: eval(f)
