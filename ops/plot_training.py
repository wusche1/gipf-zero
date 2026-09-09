"""Export self-play training measurements and an honest loss plot."""
import argparse,csv,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='*');a=p.parse_args()
    paths=[ROOT/x for x in a.runs] if a.runs else sorted((ROOT/'runs').glob('pilot_*'))+sorted((ROOT/'runs').glob('final*'))
    destination=ROOT/'reports/training';destination.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    rows=[]
    for folder in paths:
        metrics=folder/'metrics.jsonl'
        if not metrics.exists():continue
        records=[]
        for line in metrics.read_text().splitlines():
            try:
                d=json.loads(line)
                if d.get('event')=='train':records.append(d)
            except json.JSONDecodeError:continue
        if not records:continue
        shutil.copyfile(metrics,destination/f'{folder.name}-metrics.jsonl')
        if (folder/'config.json').exists():shutil.copyfile(folder/'config.json',destination/f'{folder.name}-config.json')
        for d in records:rows.append({'run':folder.name,**{k:d.get(k) for k in ['time','games','updates','cutoffs','policy_loss','value_loss']}})
        x=np.array([d['games'] for d in records]);window=min(15,max(1,len(records)//8))
        for ax,key in zip(axes,['policy_loss','value_loss']):
            y=np.array([d[key] for d in records]);smooth=np.convolve(y,np.ones(window)/window,mode='valid')
            line,=ax.plot(x[window-1:],smooth,label=folder.name.replace('pilot_',''),linewidth=1.8)
            ax.plot(x,y,alpha=.12,color=line.get_color(),linewidth=.7)
    for ax,title in zip(axes,['Policy loss (legal-action cross entropy)','Value loss (self-play outcome MSE)']):
        ax.set_title(title);ax.set_xlabel('Completed self-play games');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(fontsize=8)
    fig.suptitle('GIPF Zero training — losses describe learning targets, not playing strength',fontsize=12)
    fig.savefig(destination/'losses.png',dpi=160);fig.savefig(destination/'losses.svg');plt.close(fig)
    if rows:
        with (destination/'measurements.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'runs':len(set(x['run'] for x in rows)),'training_measurements':len(rows),'output':str(destination)}))
if __name__=='__main__':main()
