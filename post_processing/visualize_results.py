"""Read available paired metrics; building-equal summaries and publication figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

MAIN_RULES = ['Rule-%.2f'%a for a in (0,.25,.5,.75,1)]
MAIN = ['Raw'] + MAIN_RULES[1:] + [cfg.OURS]


def load_results(s):
    folder=cfg.metric_root(s);path=folder/'case_metrics.csv'
    if not path.is_file():
        print('No metrics yet:',path,'\nRun the offline apply step when the selected priors are available.')
        return None
    frame=pd.read_csv(path,keep_default_na=False,dtype={k:str for k in cfg.KEYS})
    frame=frame.loc[frame.apply(lambda r:cfg.selected(r,s),axis=1)].copy()
    if frame.empty:
        print('No completed metrics for this selection.');return None
    if frame.duplicated(cfg.KEYS+['Postprocess_Method']).any():
        raise ValueError('Duplicate case/strategy metrics')
    required=set(['Raw',cfg.OURS]+['Rule-%.2f'%a for a in s['fixed_strengths']])
    keep=[]
    for key,group in frame.groupby(cfg.KEYS,sort=False):
        if set(group.Postprocess_Method)==required and group.Points.nunique()==1 and np.isfinite(group[['RMSE','MAE','Points']].to_numpy(float)).all():
            keep.append(key)
    if not keep:
        print('No fully paired cases yet; missing cases are not filled with zero.');return None
    keys=pd.DataFrame(keep,columns=cfg.KEYS)
    paired=frame.merge(keys,on=cfg.KEYS,validate='many_to_one')
    b=paired.groupby(['Building','Train_Days','OOD_Label','Postprocess_Method'],as_index=False)[['MAE','RMSE']].mean()
    overall=b.groupby(['Train_Days','OOD_Label','Postprocess_Method'],as_index=False).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean'),Buildings=('Building','nunique'))
    return paired,b,overall


def compare_saved_reference(s, paired):
    """Compare reproduced Ours against saved pre-packaging metrics, if present."""
    if s['prior_source'] != 'existing' or s['target_matching'] != 'legacy':
        return None
    source=s['existing_priors'].parent
    parts=[]
    for name,full in [('new_case_metrics.csv',True),('cross_training_days_transfer_ood1_3/new_case_metrics.csv',False)]:
        path=source/name
        if not path.is_file():continue
        f=pd.read_csv(path,keep_default_na=False,dtype={k:str for k in cfg.KEYS})
        if not set(cfg.KEYS+['MAE','RMSE','Points']).issubset(f.columns):continue
        if 'Postprocess_Method' in f:
            f=f.loc[f.Postprocess_Method.eq(cfg.OURS)]
        f=f.loc[f.Train_Days.eq('Full') if full else ~f.Train_Days.eq('Full')]
        parts.append(f[cfg.KEYS+['MAE','RMSE','Points']])
    if not parts:
        print('No saved original metric table found at the selected source; numerical reproduction is not yet confirmed on server data.')
        return None
    reference=pd.concat(parts,ignore_index=True)
    if reference.duplicated(cfg.KEYS).any():
        print('Original reference has duplicate case identities; no automatic comparison was made.')
        return None
    ours=paired.loc[paired.Postprocess_Method.eq(cfg.OURS),cfg.KEYS+['MAE','RMSE','Points']]
    compared=ours.merge(reference,on=cfg.KEYS,suffixes=('_reproduced','_original'),validate='one_to_one')
    if compared.empty:
        print('No matching original reference cases for this selection.');return compared
    compared['Same_Points']=compared.Points_reproduced.eq(compared.Points_original)
    for metric in ('MAE','RMSE'):
        compared[metric+'_delta']=compared[metric+'_reproduced']-compared[metric+'_original']
    compared.to_csv(cfg.metric_root(s)/'figures/original_result_reproduction.csv',index=False)
    print('Original-result comparison: %d/%d selected completed cases matched.'%(len(compared),len(ours)))
    print('Same Points:',int(compared.Same_Points.sum()),'/',len(compared),
          '; max |MAE delta|:',compared.MAE_delta.abs().max(),
          '; max |RMSE delta|:',compared.RMSE_delta.abs().max())
    print('Differences require explanation before claiming original-result reproduction.')
    return compared


def report(s,make_figures=True):
    result=load_results(s)
    if result is None:return None
    paired,building,overall=result
    folder=cfg.metric_root(s);out=folder/'figures';out.mkdir(parents=True,exist_ok=True)
    planned=pd.DataFrame([r for r in cfg.read_index(s['case_index']) if cfg.selected(r,s)])
    planned=planned[cfg.KEYS].drop_duplicates()
    completed=paired[cfg.KEYS].drop_duplicates()
    keys=['Building','Train_Days','OOD_Label']
    coverage=planned.groupby(keys).size().rename('Planned').reset_index().merge(
        completed.groupby(keys).size().rename('Paired').reset_index(),on=keys,how='left')
    coverage['Paired']=coverage.Paired.fillna(0).astype(int)
    coverage['Complete']=coverage.Planned.eq(coverage.Paired)
    coverage.to_csv(out/'coverage.csv',index=False)
    overall.to_csv(out/'overall_building_equal.csv',index=False)
    building.to_csv(out/'building_metrics.csv',index=False)
    compare_saved_reference(s,paired)
    overall_table=overall.loc[overall.Postprocess_Method.isin(MAIN)].pivot(index=['Train_Days','OOD_Label'],columns='Postprocess_Method',values='MAE').reindex(columns=MAIN)
    architecture=paired.groupby(['Building','Architecture','Train_Days','OOD_Label','Postprocess_Method'],as_index=False)[['MAE','RMSE']].mean().groupby(['Architecture','Train_Days','OOD_Label','Postprocess_Method'],as_index=False)[['MAE','RMSE']].mean()
    architecture.to_csv(out/'architecture_metrics.csv',index=False)
    print('Paired cases:',len(completed),'/',len(planned),'; buildings:',completed.Building.nunique())
    print('Summary: mean case error within each building, then equal weight per building.')
    try:
        from IPython.display import display
        display(coverage);display(overall_table.round(5))
    except ImportError:
        print(coverage.to_string(index=False));print(overall_table.to_string())
    if not make_figures:return result
    import matplotlib.pyplot as plt
    oods=[o for o in s['oods'] if o in set(overall.OOD_Label)]
    ncols=min(2,len(oods));nrows=(len(oods)+ncols-1)//ncols
    label=' [PARTIAL]' if not coverage.Complete.all() else ''
    def panels():
        fig,axes=plt.subplots(nrows,ncols,figsize=(6.4*ncols,4.3*nrows),squeeze=False,constrained_layout=True)
        for ax in axes.flat[len(oods):]:ax.set_visible(False)
        return fig,list(axes.flat)
    def save(fig,name):
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight')
        plt.show();plt.close(fig)
    # Training-quantity comparison: no missing day is plotted as zero.
    fig,axes=panels()
    for ax,ood in zip(axes,oods):
        t=overall.loc[overall.OOD_Label.eq(ood)].pivot(index='Train_Days',columns='Postprocess_Method',values='MAE')
        days=[d for d in s['train_days'] if d in t.index];t=t.reindex(days)
        x=np.arange(len(days));w=.12
        for i,method in enumerate(MAIN):
            if method not in t:continue
            ax.bar(x+(i-(len(MAIN)-1)/2)*w,t[method],width=w,label=method.replace('Rule-','Fixed '))
        ax.set_xticks(x);ax.set_xticklabels(days);ax.set_ylabel('MAE (°C)');ax.set_title(ood+label)
        ax.grid(axis='y',alpha=.18)
    axes[0].legend(fontsize=8);save(fig,'training_days_MAE')
    # Full-only fixed-strength sensitivity; the test-best point is retrospective.
    if 'Full' in set(overall.Train_Days):
        fig,axes=panels()
        for ax,ood in zip(axes,oods):
            t=overall.loc[overall.OOD_Label.eq(ood)&overall.Train_Days.eq('Full')].set_index('Postprocess_Method')
            if t.empty:ax.set_visible(False);continue
            methods=[m for m in t.index if m.startswith('Rule-')];methods=sorted(methods,key=lambda m:float(m[5:]))
            ax.plot([float(m[5:]) for m in methods],t.loc[methods,'MAE'],'o-',ms=3,color='gray',label='Fixed strengths')
            ax.axhline(t.loc[cfg.OURS,'MAE'],color='#2477ad',ls='--',label='Ours')
            ax.set_title(ood+' / Full'+label);ax.set_xlabel('Gate');ax.set_ylabel('MAE (°C)');ax.grid(alpha=.18)
        axes[0].legend(fontsize=8);save(fig,'fixed_strength_sensitivity_MAE')
        fig,axes=panels()
        heat_values=[]
        for ood in oods:
            t=building.loc[building.OOD_Label.eq(ood)&building.Train_Days.eq('Full')].pivot(index='Building',columns='Postprocess_Method',values='MAE')
            refs=[m for m in MAIN_RULES if m in t]
            if t.empty or not refs or cfg.OURS not in t:
                heat_values.append(pd.DataFrame());continue
            heat_values.append(100*t[refs].subtract(t[cfg.OURS],axis=0).divide(t[refs].replace(0,np.nan)))
        finite=np.concatenate([x.to_numpy().ravel() for x in heat_values]);finite=finite[np.isfinite(finite)]
        lim=max(1,float(np.max(np.abs(finite)))) if len(finite) else 1
        for ax,ood,gain in zip(axes,oods,heat_values):
            if gain.empty:ax.set_visible(False);continue
            image=ax.imshow(gain.to_numpy(),cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
            ax.set_yticks(range(len(gain)));ax.set_yticklabels(gain.index,fontsize=8)
            ax.set_xticks(range(len(gain.columns)));ax.set_xticklabels([m.replace('Rule-','') for m in gain.columns])
            ax.set_title(ood+' / Full'+label);ax.set_xlabel('Fixed gate baseline')
            for y in range(len(gain)):
                for x in range(len(gain.columns)):
                    v=gain.iloc[y,x]
                    if np.isfinite(v):ax.text(x,y,'%+.1f'%v,ha='center',va='center',fontsize=8)
            gain.to_csv(out/(ood+'_building_gains_MAE.csv'))
        if any(not x.empty for x in heat_values):
            fig.colorbar(image,ax=[a for a in axes[:len(oods)] if a.get_visible()],label='MAE reduction (%)')
        save(fig,'building_gains_MAE')
    print('Figures and tables:',out)
    return result


if __name__=='__main__':
    settings,_,_=cfg.cli()
    report(settings)
