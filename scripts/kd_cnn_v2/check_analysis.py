"""Focused independent tie/strict-cut checks for reported quality metrics."""
import argparse,json
from pathlib import Path
import numpy as np
from analysis import auc,roc,fit_threshold


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    rng=np.random.default_rng(7);checks=[]
    for n in (4,11,100):
        for _ in range(10):
            scores=rng.integers(-4,5,n);labels=rng.integers(0,2,n);labels[:2]=[0,1]
            pos,neg=scores[labels==1],scores[labels==0]
            brute=float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean())
            curve=roc(scores,labels)
            area=sum((b['fpr']-a['fpr'])*(a['tpr']+b['tpr'])/2 for a,b in zip(curve[:-1],curve[1:]))
            if abs(auc(scores,labels)-brute)>1e-12 or abs(area-brute)>1e-12:raise AssertionError('AUC/tied ROC mismatch')
            for alpha in (.005,.01,.02,.05):
                threshold=fit_threshold(neg,alpha)
                actual=int(np.sum(neg>threshold))
                if actual>int(alpha*len(neg)+1e-10):raise AssertionError('Strict threshold budget exceeded')
            checks.append({'n':n,'auc':brute,'roc_area':area})
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'passed':True,'independent_cases':len(checks),'checks':checks},indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'independent_cases':len(checks)}))


if __name__=='__main__':main()
