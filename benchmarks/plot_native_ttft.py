"""Plot all twelve pre-registered TTFT pairs; no GPU or model execution."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise RuntimeError('preserve existing figure')
    r=json.loads(a.suite.read_text());s=r['performance_summary']
    if r['status']!='passed' or len(s['pairs'])!=12:raise RuntimeError('incomplete suite')
    cases=['archive-T16480','archive-T32864','code-T16480','code-T32864']
    fig,(ax,ratio)=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.2,1]})
    x=list(range(4));medians=[]
    for arm,color,offset,label in [('baseline','#6b7280',-.18,'Baseline'),('direct_store','#087f8c',.18,'Direct KV store')]:
        values=[[row['ttft_ms_'+arm]/1000 for row in s['pairs'] if row['case']==name] for name in cases]
        heights=[statistics.median(v) for v in values]
        ax.bar([i+offset for i in x],heights,width=.34,color=color,label=label)
        for i,vals in enumerate(values):ax.scatter([i+offset]*len(vals),vals,color='black',s=12,zorder=3)
    ax.set_xticks(x,['Archive\n16K','Archive\n32K','Code\n16K','Code\n32K']);ax.set_ylabel('Warm encoded-request TTFT (s)')
    ax.set_ylim(0,13.4);ax.legend(frameon=False,fontsize=9);ax.set_title('Median bars; every process shown')
    for group,color in enumerate(['#2563eb','#d97706','#7c3aed']):
        vals=[next(row['ttft_ms_ratio'] for row in s['pairs'] if row['case']==name and row['group']==group) for name in cases]
        ratio.scatter(vals,[i+(group-1)*.13 for i in x],s=38,color=color,label=['A/B','B/A','A/B'][group]+f' group {group+1}')
    gm=s['candidate_over_baseline']['ttft_ms']['all_12']['geomean']
    ratio.axvline(1,color='#6b7280',linestyle='--',linewidth=1);ratio.axvline(.97,color='#087f8c',linestyle=':',linewidth=1)
    ratio.set_yticks(x,cases);ratio.invert_yaxis();ratio.set_xlim(.93,1.005)
    ratio.set_xlabel('Candidate / baseline TTFT (lower is faster)');ratio.set_title(f'All 12 paired inputs: ratio {gm:.4f}')
    ratio.legend(frameon=False,fontsize=8,loc='lower right');ratio.grid(axis='x',alpha=.2)
    fig.suptitle('NanoKV M23 · fixed sparse policy · Qwen3-4B · RTX 3080 Laptop',fontsize=12)
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)
    plt.close(fig);print(a.output)


if __name__=='__main__':run()
