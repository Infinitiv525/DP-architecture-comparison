"""Small standalone scientific plots from the archived numeric cost curves."""
import csv
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
if (HERE/'.plot_deps').exists():sys.path.insert(0,str(HERE/'.plot_deps'))
os.environ['MPLCONFIGDIR']=str(HERE/'.plot_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    report=HERE/'report/continuation'
    summary=json.loads((report/'summary/complete.json').read_text())
    parents=json.loads((HERE/'parents.json').read_text())['tasks']
    with (report/'summary/cost_curves.csv').open(newline='') as f:curves=list(csv.DictReader(f))
    assert len(summary['models'])==4
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharex=True,sharey=True)
    colors=['#1f77b4','#d97706']
    for ax,family,label in zip(axes,['chebkan','rbf'],['ChebKAN','RBF']):
        ax.axhline(100,color='#7b8794',ls='--',lw=1.1)
        for i,seed in enumerate([931101,931102]):
            model=next(r for r in summary['models'] if r['architecture']==family and r['seed']==seed)
            parent=next(r for r in parents if r['model']==family and r['seed']==seed)
            values=[r for r in curves if r['id']==model['id'] and r['phase']=='exploratory_extra4h']
            baseline=parent['original_best_VAL_loss'];assert baseline>0 and values
            x=[parent['original_allocated_GPU_seconds']/3600]+[float(r['cumulative_GPU_h']) for r in values]
            y=[100.]+[100*float(r['best_so_far_VAL_loss'])/baseline for r in values]
            assert all(b>=a for a,b in zip(x,x[1:]))
            ax.step(x,y,where='post',color=colors[i],lw=2,label='seed '+str(seed))
            done=json.loads((report/'continuation'/model['id']/'complete.json').read_text())
            ax.scatter([model['cumulative_GPU_h']],[100*done['last_VAL_loss']/baseline],marker='x',s=60,
                       color=colors[i],linewidths=2,zorder=4)
        ax.set_title(label,fontweight='bold',pad=10)
        ax.set_xlabel('Celkový pozorovaný čas na H200 [GPU-h]')
        ax.set_xlim(3.85,8.0);ax.set_ylim(80,102)
        ax.set_xticks([4,5,6,7,8]);ax.set_yticks([80,85,90,95,100])
        ax.grid(color='#d8dee5',alpha=.8,lw=.65)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower left',frameon=False)
    axes[0].set_ylabel('Najlepšia VAL strata / pôvodná najlepšia [%]')
    fig.suptitle('Druhá šanca: mierny zisk za ďalšie štyri GPU hodiny',fontweight='bold',fontsize=14,y=.985)
    fig.text(.5,.91,'Dva vopred určené seedy na rodinu · samostatný exploratívny doplnok',ha='center',color='#52606d')
    fig.text(.5,.025,'100 % = pôvodný 4h výsledok; nižšie je lepšie.  × = posledný stav, nie vybraný najlepší model.',ha='center',fontsize=9,color='#52606d')
    fig.subplots_adjust(left=.08,right=.99,bottom=.17,top=.80,wspace=.16)
    folder=HERE/'figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','svg']:
        path=folder/('druha_sanca_priebeh.'+suffix)
        fig.savefig(path,dpi=170,facecolor='white')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    (HERE/'plotting_environment.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__,numpy=np.__version__,
        data_source='report/continuation/summary/cost_curves.csv',
        note='Step curve is best-so-far GPU VAL normalized by original GPU baseline; x marker is actual last checkpoint. No raw fields or weights used.'),indent=2)+'\n',encoding='utf-8')
    print('PNG and editable SVG written:',folder)


if __name__=='__main__':main()
