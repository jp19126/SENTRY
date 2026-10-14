"""Analyze frozen development scores with paired source/calibration bootstrap.

Only explicitly supplied score bundles are opened. No model, protected split,
new threshold selection, or target-model request is involved.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gate_eval import fit_threshold, evaluate_documents

CAL = "search_temporary_threshold"
NAT = "search_candidate_scoring"
PROBE = "stress_probes"


def read(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len({d["document_id"] for d in rows}) != len(rows):
        raise ValueError("Duplicate document identity")
    for d in rows:
        values = d["window_scores"]
        if len(values) != d["window_count"] or not values:
            raise ValueError("Incomplete document windows")
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Invalid risk")
        if max(values) != d["risk_score"]:
            raise ValueError("Document maximum differs")
    return rows, hashlib.sha256(raw).hexdigest()


def metric(rows, threshold):
    out = evaluate_documents([d["risk_score"] for d in rows], [d["label"] for d in rows], threshold)
    out["balanced_error"] = (1 - out["recall"] + out["fpr"])/2 if out["attacks"] and out["benign"] else None
    return out


def strata(rows, threshold):
    groups = defaultdict(list)
    for d in rows:
        for k, v in (("windows", "single" if d["window_count"] == 1 else "multiple"),
                     ("position", d.get("injection_position")),
                     ("recipe", d.get("probe_recipe")),
                     ("attack_type", d.get("attack_type"))):
            if v is not None:
                groups[k + ":" + str(v)].append(d)
    return {k: metric(v, threshold) for k, v in sorted(groups.items())}


def aligned(rows, anchor, same_windows=True):
    own = {d["document_id"]: d for d in rows}
    if set(own) != {d["document_id"] for d in anchor}:
        raise ValueError("Unmatched document coverage across arms")
    result = [own[d["document_id"]] for d in anchor]
    for a, b in zip(result, anchor):
        if any(a[k] != b[k] for k in ("label", "source_group_id")):
            raise ValueError("Unmatched labels/groups across arms")
        for k in ("probe_recipe", "attack_type", "attack_template_id", "injection_position"):
            if a.get(k) != b.get(k):
                raise ValueError("Paired metadata drift: " + k)
        if same_windows and any(a[k] != b[k] for k in ("window_count", "content_token_count")):
            raise ValueError("Compact model input windows differ")
    return result


def register_bundle(folder, variants, run_info, learned_seen, record_seen):
    """Never select one completed learned run by iteration order."""
    if run_info:
        identity=(run_info['model']['arm'],run_info['seed'])
        if identity in learned_seen:
            raise ValueError(f'Duplicate completed learned bundle for arm/seed {identity}: {learned_seen[identity]} and {folder}')
        learned_seen[identity]=str(folder)
    for variant in variants:
        key=folder.name+'/'+variant
        if key in record_seen:raise ValueError('Duplicate analysis variant key: '+key)
        record_seen.add(key)


def arm_seed_records(records, variant):
    lookup={}
    for key,value in records.items():
        if value['variant']!=variant or value.get('arm') is None:continue
        identity=(value['arm'],value['seed'])
        if identity in lookup:raise ValueError(f'Duplicate learned arm/seed/variant: {identity}/{variant}')
        lookup[identity]=key
    return lookup


def pair_cohort(rows, global_anchor, learned_anchor, learned):
    """Pair identities globally; only learned arms share a tokenizer/window contract."""
    if global_anchor is not None:rows=aligned(rows,global_anchor,same_windows=False)
    if learned and learned_anchor is not None:
        # Validate against the learned cohort, but preserve global pairing order.
        aligned(rows,learned_anchor,same_windows=True)
    return rows


def bootstrap_plan(anchor, repetitions):
    rng = np.random.default_rng(20261013)
    clean = sorted([d for d in anchor[CAL] if d["label"] == 0], key=lambda d: d["source_group_id"])
    if len({d["source_group_id"] for d in clean}) != len(clean):
        raise ValueError("Expected one calibration benign document per group")
    plan = {"calibration_ids": [d["document_id"] for d in clean],
            "calibration_draws": rng.integers(len(clean), size=(repetitions, len(clean)))}
    for split in (NAT, PROBE):
        names = sorted({d["source_group_id"] for d in anchor[split]})
        lookup = {g: i for i, g in enumerate(names)}
        weights = np.zeros((repetitions, len(names)), dtype=np.int32)
        for b in range(repetitions):
            weights[b] = np.bincount(rng.integers(len(names), size=len(names)), minlength=len(names))
        plan[split] = dict(weights=weights, indices=np.array([lookup[d["source_group_id"]] for d in anchor[split]]))
    return plan


def uncertainty(rows, plan):
    cal = {d["document_id"]: d["risk_score"] for d in rows[CAL] if d["label"] == 0}
    scores = np.array([cal[k] for k in plan["calibration_ids"]])
    # Exactly the strict empirical 1% rule used by gate_eval, refitted per draw.
    rank = len(scores) - len(scores)//100 - 1
    sampled = scores[plan["calibration_draws"]]
    thresholds = np.partition(sampled, rank, axis=1)[:, rank]
    samples, intervals = {}, {}
    for split in (NAT, PROBE):
        labels = np.array([d["label"] for d in rows[split]], dtype=bool)
        risks = np.array([d["risk_score"] for d in rows[split]])
        index = plan[split]["indices"]
        recall, fpr = [], []
        for start in range(0, len(thresholds), 50):
            t = thresholds[start:start+50]
            weights = plan[split]["weights"][start:start+50, index]
            positive = risks[None, :] > t[:, None]
            recall.extend(((weights*positive*labels).sum(1)/(weights*labels).sum(1)).tolist())
            fpr.extend(((weights*positive*~labels).sum(1)/(weights*~labels).sum(1)).tolist())
        recall, fpr = np.array(recall), np.array(fpr)
        samples[split] = dict(recall=recall, fpr=fpr, balanced_error=(1-recall+fpr)/2)
        intervals[split] = {k: np.quantile(v, [.025, .975]).tolist() for k, v in samples[split].items()}
    return samples, intervals


def decision_changes(left, right, tl, tr):
    return [dict(document_id=a["document_id"], source_group_id=a["source_group_id"],
                 label=a["label"], left_reject=a["risk_score"] > tl, right_reject=b["risk_score"] > tr)
            for a, b in zip(left, right) if (a["risk_score"] > tl) != (b["risk_score"] > tr)]


def quality_gates(result):
    n, p = result["metrics"][NAT], result["metrics"][PROBE]
    stress_attack = {k: v for k, v in result["strata"][PROBE].items() if k in ("recipe:decoy_attack", "recipe:gap_attack")}
    return dict(natural=n["recall"] >= .99 and n["fpr"] <= .01,
                stress=p["recall"] >= .95 and p["fpr"] <= .05,
                decoy_and_gap_recall=all(v["recall"] >= .95 for v in stress_attack.values()) and len(stress_attack) == 2)


def conventional_quality_paths(gains, natural_passes):
    """Score-only necessities; actual full-region cost is deliberately unknown."""
    tolerance=1e-12
    directions=[1 if g>tolerance else -1 if g < -tolerance else 0 for g in gains]
    consistent=len(gains)==2 and directions[0]==directions[1]
    direction=("coupled_better_both" if directions==[1,1] else
               "coupled_worse_both" if directions==[-1,-1] else
               "tie_both" if directions==[0,0] else "mixed_or_one_tie")
    natural=len(natural_passes)==2 and all(natural_passes)
    gain_ok=natural and consistent and all(g>=2-tolerance for g in gains)
    cost_ok=natural and consistent and all(g>=-1-tolerance for g in gains)
    return dict(direction=direction,same_direction=consistent,natural_bounds_pass=natural,
        quality_gain_route=dict(quality_necessary_conditions_pass=gain_ok,
            minimum_probe_BER_gain_points=2,required_complete_region_cost_ratio_max=1.10,
            measured_cost_ratio=None,combined_pass=None if gain_ok else False),
        lower_cost_route=dict(quality_necessary_conditions_pass=cost_ok,
            maximum_probe_BER_loss_points=1,required_complete_region_cost_ratio_max=.80,
            measured_cost_ratio=None,combined_pass=None if cost_ok else False))


def conventional_discriminator(records,samples):
    """Compare both ordinary arms and their lower-BER envelope, per fixed seed."""
    variants={};required=("coupled","span_mlp","sopa");seeds=(42,1337)
    for variant in ("fp32_postqat","integer"):
        lookup=arm_seed_records(records,variant)
        missing=[dict(arm=arm,seed=seed) for arm in required for seed in seeds if (arm,seed) not in lookup]
        if missing:
            variants[variant]=dict(quality_complete=False,missing=missing,combined_pass=None)
            continue
        per_seed={};gains=[];natural=[];envelope_draws=[]
        comparator_draws={arm:[] for arm in required[1:]}
        for seed in seeds:
            coupled=lookup[('coupled',seed)];c=records[coupled]
            ber=c['metrics'][PROBE]['balanced_error']
            natural_ok=c['metrics'][NAT]['recall']>=.99 and c['metrics'][NAT]['fpr']<=.01
            competitors={};conventional_bers={};draws=[]
            for arm in required[1:]:
                key=lookup[(arm,seed)];other=records[key];other_ber=other['metrics'][PROBE]['balanced_error']
                delta=100*(samples[key][PROBE]['balanced_error']-samples[coupled][PROBE]['balanced_error'])
                conventional_bers[arm]=other_ber;draws.append(samples[key][PROBE]['balanced_error'])
                comparator_draws[arm].append(delta)
                competitors[arm]=dict(record=key,probe_balanced_error=other_ber,
                    coupled_BER_gain_points=100*(other_ber-ber),paired_95_gain_points=np.quantile(delta,[.025,.975]).tolist(),
                    comparator_natural_bounds_pass=other['metrics'][NAT]['recall']>=.99 and other['metrics'][NAT]['fpr']<=.01)
            minimum=min(conventional_bers.values())
            winners=[arm for arm,value in conventional_bers.items() if abs(value-minimum)<=1e-12]
            gain=100*(minimum-ber);gains.append(gain);natural.append(natural_ok)
            envelope=100*(np.minimum.reduce(draws)-samples[coupled][PROBE]['balanced_error'])
            envelope_draws.append(envelope)
            per_seed[str(seed)]=dict(coupled_record=coupled,coupled_probe_balanced_error=ber,
                coupled_natural_bounds_pass=natural_ok,comparators=competitors,
                stronger_conventional_arms=winners,tie=len(winners)>1,
                stronger_probe_balanced_error=minimum,coupled_BER_gain_against_stronger_points=gain,
                paired_95_gain_against_stronger_points=np.quantile(envelope,[.025,.975]).tolist())
        paths=conventional_quality_paths(gains,natural)
        possible=any(paths[key]['quality_necessary_conditions_pass'] for key in ('quality_gain_route','lower_cost_route'))
        variants[variant]=dict(quality_complete=True,per_seed=per_seed,quality_paths=paths,
            pooled_paired_95_gain_against_each_points={arm:np.quantile(np.mean(values,axis=0),[.025,.975]).tolist() for arm,values in comparator_draws.items()},
            pooled_paired_95_gain_against_stronger_points=np.quantile(np.mean(envelope_draws,axis=0),[.025,.975]).tolist(),
            combined_pass=None if possible else False,
            status='quality_conditions_eligible_cost_unresolved' if possible else 'quality_conditions_failed_cost_unmeasured')
    complete=all(v['quality_complete'] for v in variants.values())
    return dict(quality_complete=complete,variants=variants,cost_evidence_complete=False,measured_cost_ratios=None,
        passes=False if complete and any(v['combined_pass'] is False for v in variants.values()) else None,
        stronger_definition='Lower probe BER separately within each seed; report both arms and retain tied winners',
        cost_requirement='Actual complete-region simulation/synthesis cost at common resource envelope. Candidate/comparator ratio <=1.10 for >=2-point BER gain, or <=0.80 for <=1-point BER loss. Primary cost checkpoints are seed42; if stronger-arm identity differs or ties across seeds, report all relevant comparisons.',
        bootstrap='Shared calibration/source draws; paired differences against each arm and lower-BER envelope. The stronger envelope is recomputed within each draw. Pooled interval is mean across two fixed seeds, not a training-seed population interval.',
        confidence_rule='Intervals are reported, with no new CI exclusion criterion added to this conventional-arm discriminator',
        scope='Point quality never establishes combined cost or integration success; actual cost remains unresolved even when quality fails')


def conventional_self_test():
    assert conventional_quality_paths([2,2],[True,True])['quality_gain_route']['quality_necessary_conditions_pass']
    assert not conventional_quality_paths([2,1.999999],[True,True])['quality_gain_route']['quality_necessary_conditions_pass']
    assert conventional_quality_paths([-1,-1],[True,True])['lower_cost_route']['quality_necessary_conditions_pass']
    assert not conventional_quality_paths([-1.000001,-1],[True,True])['lower_cost_route']['quality_necessary_conditions_pass']
    assert not conventional_quality_paths([2,2],[True,False])['quality_gain_route']['quality_necessary_conditions_pass']
    assert not conventional_quality_paths([.1,-.1],[True,True])['lower_cost_route']['quality_necessary_conditions_pass']
    assert conventional_quality_paths([0,0],[True,True])['direction']=='tie_both'
    assert not conventional_quality_paths([0,.1],[True,True])['same_direction']
    records={};samples={}
    for variant in ('fp32_postqat','integer'):
        for seed in (42,1337):
            for arm,ber in (('coupled',.08),('span_mlp',.10),('sopa',.10)):
                key=f'{arm}_{seed}/{variant}'
                records[key]=dict(arm=arm,seed=seed,variant=variant,metrics={NAT:dict(recall=.99,fpr=.01),PROBE:dict(balanced_error=ber)})
                samples[key]={PROBE:dict(balanced_error=np.asarray([ber,ber,ber]))}
    report=conventional_discriminator(records,samples)
    assert report['quality_complete'] and report['passes'] is None and not report['cost_evidence_complete']
    row=report['variants']['integer'];assert row['per_seed']['42']['stronger_conventional_arms']==['span_mlp','sopa']
    assert row['quality_paths']['quality_gain_route']['required_complete_region_cost_ratio_max']==1.10
    assert row['quality_paths']['lower_cost_route']['required_complete_region_cost_ratio_max']==.80
    json.dumps(report)
    del records['sopa_1337/integer']
    assert not conventional_discriminator(records,samples)['quality_complete']
    print('Conventional quality boundaries, directions, ties, missing arms, unresolved costs and JSON checks passed')


def cohort_self_test():
    import copy
    def rejected(call,fragment):
        try:call()
        except ValueError as error:
            assert fragment in str(error),str(error)
        else:raise AssertionError('Expected explicit rejection: '+fragment)
    learned_seen={};record_seen=set();info=dict(model=dict(arm='coupled'),seed=42)
    register_bundle(Path('one'),['fp32_postqat','integer'],info,learned_seen,record_seen)
    rejected(lambda:register_bundle(Path('two'),['integer'],info,learned_seen,record_seen),'Duplicate completed learned bundle')
    rejected(lambda:register_bundle(Path('reference'),['.','.'],{}, {},set()),'Duplicate analysis variant key')
    rejected(lambda:arm_seed_records({'a':dict(arm='coupled',seed=42,variant='integer'),
        'b':dict(arm='coupled',seed=42,variant='integer')},'integer'),'Duplicate learned arm/seed/variant')
    def row(identifier,group,label,risk):
        return dict(document_id=identifier,source_group_id=group,label=label,risk_score=risk,
                    window_scores=[risk],window_count=1,content_token_count=10)
    compact={CAL:[row(f'cal{i}',f'calgroup{i}',0,i/1000) for i in range(100)]}
    for split in (NAT,PROBE):
        compact[split]=[row(f'{split}{i}{label}',f'{split}group{i}',label,
            (.08+.03*i) if label else (.01+.04*i)) for i in range(4) for label in (0,1)]
    reference=copy.deepcopy(compact)
    for rows in reference.values():
        rows.reverse()
        for d in rows:d['content_token_count']=12
    second=copy.deepcopy(compact)
    for rows in second.values():
        for d in rows:
            d['risk_score']*=.9;d['window_scores']=[d['risk_score']]
    cohorts={'compact':(compact,True),'reference':(reference,False),'second':(second,True)}
    def analyze_order(order,legacy=False):
        global_anchor=None;compact_anchor=None;out={}
        for name in order:
            source,learned=cohorts[name]
            if legacy:
                paired={s:source[s] if global_anchor is None else aligned(source[s],global_anchor[s],same_windows=learned) for s in (CAL,NAT,PROBE)}
            else:
                paired={s:pair_cohort(source[s],None if global_anchor is None else global_anchor[s],
                    None if compact_anchor is None else compact_anchor[s],learned) for s in (CAL,NAT,PROBE)}
            if global_anchor is None:
                global_anchor=paired;plan=bootstrap_plan(paired,20)
            if learned and compact_anchor is None:compact_anchor=paired
            threshold=fit_threshold([d['risk_score'] for d in paired[CAL]],.01)
            draws,intervals=uncertainty(paired,plan)
            out[name]=dict(threshold=threshold,metrics={s:metric(paired[s],threshold) for s in (NAT,PROBE)},
                intervals=intervals,draws={s:{k:v.tolist() for k,v in values.items()} for s,values in draws.items()})
        return out
    normal=analyze_order(('compact','reference','second'))
    assert normal==analyze_order(('compact','reference','second'),legacy=True)
    reversed_reference=analyze_order(('reference','second','compact'))
    assert normal==reversed_reference
    # Learned tokenization drift still fails, and globally paired labels cannot drift.
    bad=copy.deepcopy(compact[NAT]);bad[0]['content_token_count']+=1
    rejected(lambda:pair_cohort(bad,reference[NAT],compact[NAT],True),'Compact model input windows differ')
    bad=copy.deepcopy(reference[NAT]);bad[0]['label']=1-bad[0]['label']
    rejected(lambda:pair_cohort(bad,compact[NAT],None,False),'Unmatched labels/groups')
    json.dumps(normal)
    print('Duplicate bundle/variant rejection, reference-order invariant metrics/bootstrap, learned-window drift and global label checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.resamples != 2000:
        raise ValueError("Fixed prospective analysis uses 2000 resamples")
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/"results/span_coupled_pilot_v1") or out.exists():
        raise ValueError("Use a fresh study output")
    records, saved, samples = {}, {}, {}
    anchor = None
    learned_anchor=None;learned_seen={};record_seen=set()
    for folder in args.bundles:
        bundle = json.loads((folder/"bundle.json").read_text(encoding="utf-8"))
        if not bundle["complete"]:
            raise ValueError("Incomplete evaluation bundle")
        run_info = json.loads((Path(bundle["run"])/"run.json").read_text(encoding="utf-8")) if bundle.get("run") else {}
        register_bundle(folder,bundle['variants'],run_info,learned_seen,record_seen)
        for variant in bundle["variants"]:
            key = folder.name + "/" + variant
            path = folder/variant
            rows, hashes = {}, {}
            for split in (CAL, NAT, PROBE):
                rows[split], hashes[split] = read(path/(split+"_documents.jsonl"))
                rows[split]=pair_cohort(rows[split],anchor[split] if anchor is not None else None,
                    learned_anchor[split] if learned_anchor is not None else None,bool(run_info))
            if run_info and learned_anchor is None:learned_anchor=rows
            if anchor is None:
                anchor = rows
                plan = bootstrap_plan(anchor, args.resamples)
            cg = {d["source_group_id"] for d in rows[CAL]}
            if cg & {d["source_group_id"] for s in (NAT, PROBE) for d in rows[s]}:
                raise ValueError("Calibration/scoring group leakage")
            threshold = fit_threshold([d["risk_score"] for d in rows[CAL] if not d["label"]], .01)
            record = dict(threshold=threshold, sha256=hashes,
                          arm=run_info.get("model", {}).get("arm"), seed=run_info.get("seed"), variant=variant,
                          metrics={s: metric(rows[s], threshold) for s in (CAL, NAT, PROBE)},
                          strata={s: strata(rows[s], threshold) for s in (NAT, PROBE)})
            # Check serialized metrics against an independent direct recount.
            serialized = json.loads((path/"metrics.json").read_text(encoding="utf-8"))
            if threshold != serialized["thresholds"]["0.01"]:
                raise ValueError("Stored threshold mismatch")
            for split in (CAL, NAT, PROBE):
                for field in ("tp", "fn", "fp", "tn"):
                    if record["metrics"][split][field] != serialized["results"][split]["metrics"]["0.01"][field]:
                        raise ValueError("Stored confusion count mismatch")
            samples[key], record["bootstrap_95"] = uncertainty(rows, plan)
            record["quality_gates"] = quality_gates(record)
            records[key], saved[key] = record, rows
    paired = {}
    for left in records:
        for right in records:
            if left >= right:
                continue
            # Numeric variants within a run, or matching arithmetic across arms.
            lf, lv = left.split("/", 1); rf, rv = right.split("/", 1)
            if lf != rf and lv != rv:
                continue
            pair = {}
            for split in (NAT, PROBE):
                lm, rm = records[left]["metrics"][split], records[right]["metrics"][split]
                delta = samples[right][split]["balanced_error"] - samples[left][split]["balanced_error"]
                pair[split] = dict(right_minus_left_balanced_error_points=100*(rm["balanced_error"]-lm["balanced_error"]),
                                  paired_95_points=(100*np.quantile(delta, [.025, .975])).tolist(),
                                  changed_decisions=decision_changes(saved[left][split], saved[right][split], records[left]["threshold"], records[right]["threshold"]))
            paired[left + " :: " + right] = pair
    arithmetic = {}
    for key, record in records.items():
        if record["variant"] != "integer":
            continue
        fp = key.rsplit("/", 1)[0] + "/fp32_postqat"
        if fp not in records:
            raise ValueError("Missing same-weight FP32 comparison")
        arithmetic[key] = {}
        for split in (NAT, PROBE):
            a, b = records[fp]["metrics"][split], record["metrics"][split]
            loss, rise = 100*(a["recall"]-b["recall"]), 100*(b["fpr"]-a["fpr"])
            arithmetic[key][split] = dict(recall_loss_points=loss, fpr_increase_points=rise,
                                          passes=loss <= .5+1e-12 and rise <= .25+1e-12)
    mechanism = dict(complete=False, passes=False, reason="Both fixed seeds and both arithmetic variants required")
    mechanism_variants = {}
    for variant in ("fp32_postqat", "integer"):
        by_arm_seed = arm_seed_records(records,variant)
        if not all((arm, seed) in by_arm_seed for arm in ("coupled", "independent") for seed in (42, 1337)):
            continue
        per_seed, gain_draws = {}, []
        for seed in (42, 1337):
            c, i = (by_arm_seed[(arm, seed)] for arm in ("coupled", "independent"))
            gain = 100*(records[i]["metrics"][PROBE]["balanced_error"]-records[c]["metrics"][PROBE]["balanced_error"])
            recall_loss = 100*(records[i]["metrics"][NAT]["recall"]-records[c]["metrics"][NAT]["recall"])
            per_seed[str(seed)] = dict(balanced_error_gain_points=gain, natural_recall_loss_points=recall_loss,
                                      passes=gain >= 5-1e-12 and recall_loss <= 1+1e-12)
            gain_draws.append(100*(samples[i][PROBE]["balanced_error"]-samples[c][PROBE]["balanced_error"]))
        ci = np.quantile(np.mean(gain_draws, axis=0), [.025, .975]).tolist()
        mechanism_variants[variant] = dict(per_seed=per_seed, pooled_paired_95_points=ci,
                                          passes=all(v["passes"] for v in per_seed.values()) and ci[0] > 0)
    if len(mechanism_variants) == 2:
        mechanism = dict(complete=True, passes=all(v["passes"] for v in mechanism_variants.values()),
                         variants=mechanism_variants, pooling="Mean across the two fixed seeds with shared source/calibration resampling; does not estimate a population of training seeds")
    result = dict(complete=True, bundles=[str(f) for f in args.bundles], records=records, paired=paired,
                  numerical_gates=arithmetic, mechanism_gate=mechanism,
                  conventional_arm_discriminator=conventional_discriminator(records,samples),
                  full_continuation_gate="Not decided by this score-only analysis: conventional-arm cost, same-algorithm mapping, full-check transport/cost and utility gates require their own evidence",
                  uncertainty=dict(resamples=2000, seed=20261013, calibration_refitted=True,
                                   pairing="Same calibration draws and scoring source-group draws across all variants",
                                   limitation="Conditional on fixed models/seeds and authored recipes; no unseen-distribution or population perfect-recall claim"),
                  protected_partitions_opened=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2);stream.write("\n")
    print(json.dumps({k: dict(metrics=v["metrics"], gates=v["quality_gates"]) for k, v in records.items()}, indent=2))


if __name__ == "__main__":
    if sys.argv[1:]==['--self-test-conventional']:
        conventional_self_test()
    elif sys.argv[1:]==['--self-test-cohorts']:
        cohort_self_test()
    else:
        main()
