"""Separate descriptive output: cost curves and all four exploratory outcomes."""
import argparse
import csv
from pathlib import Path
import statistics
from runtime import read,dependencies


def main(root):
    policy,parents,common,_,_=dependencies();models=[];curves=[];full=[]
    for row in parents['tasks']:
        audit=read(root/'audit'/row['id']/'complete.json')
        assert audit['status']=='EXPLORATORY_SECOND_CHANCE_CPU_AUDIT_PASS'
        assert audit['no_TEST_targets'] and audit['excluded_from_primary_experiment']
        done=read(root/'continuation'/row['id']/'complete.json')
        original=read(row['original_complete']);metrics=audit['metrics'];chosen=audit['selected_best_so_far']
        r=dict(id=row['id'],architecture=row['model'],seed=row['seed'],
            baseline_VAL_loss=metrics['baseline_4h']['phase_metrics']['validation']['mean_raw_case_balanced_loss'],
            continuation_best_VAL_loss=metrics['continuation_best']['phase_metrics']['validation']['mean_raw_case_balanced_loss'],
            continuation_last_VAL_loss=metrics['continuation_last']['phase_metrics']['validation']['mean_raw_case_balanced_loss'],
            selected_VAL_loss=metrics[chosen]['phase_metrics']['validation']['mean_raw_case_balanced_loss'],
            selected_reduction_percent=audit['selected_reduction_percent'],
            phase_best_reduction_percent=audit['phase_best_reduction_percent'],phase_last_reduction_percent=audit['phase_last_reduction_percent'],
            baseline_velocity_L2=metrics['baseline_4h']['physical_validation']['mean_velocity_relative_L2'],
            baseline_pressure_L2=metrics['baseline_4h']['physical_validation']['mean_pressure_relative_L2'],
            selected_velocity_L2=metrics[chosen]['physical_validation']['mean_velocity_relative_L2'],
            selected_pressure_L2=metrics[chosen]['physical_validation']['mean_pressure_relative_L2'],
            original_GPU_h=audit['original_GPU_seconds']/3600,additional_GPU_h=audit['additional_GPU_seconds']/3600,
            cumulative_GPU_h=audit['cumulative_observed_GPU_seconds']/3600,selected_is_original=chosen=='baseline_4h',
            excluded_from_primary_experiment=True)
        models.append(r);full.append(audit)
        best=float('inf')
        for h in original['history']:
            score=h['validation']['mean_raw_case_balanced_loss'];best=min(best,score)
            curves.append(dict(id=row['id'],architecture=row['model'],seed=row['seed'],phase='original_4h',
                cumulative_GPU_h=h['allocation_elapsed_seconds']/3600,VAL_loss=score,best_so_far_VAL_loss=best))
        for h in done['history']:
            score=h['validation']['mean_raw_case_balanced_loss'];best=min(best,score)
            curves.append(dict(id=row['id'],architecture=row['model'],seed=row['seed'],phase='exploratory_extra4h',
                cumulative_GPU_h=h['cumulative_allocated_GPU_seconds']/3600,VAL_loss=score,best_so_far_VAL_loss=best))
    assert len(models)==4
    out=root/'summary';out.mkdir(parents=True,exist_ok=False)
    for name,values in [('model_metrics.csv',models),('cost_curves.csv',curves)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    common.save(out/'complete.json',dict(status='SEPARATE_SECOND_CHANCE_COMPLETE',models=models,full_audits=full,
        total_additional_observed_GPU_hours=sum(r['additional_GPU_h'] for r in models),
        no_TEST_targets=True,excluded_from_primary_experiment=True,repetitions_per_family=2,
        nominal_budget_per_lineage_GPU_h=8,additional_16h_extension_not_scheduled=True))
    text=['DRUHA SANCA DOKONCENA - SAMOSTATNY EXPLORATIVNY DOPLNOK','',
          'ChebKAN a RBF: po dva vopred urcene seedy, pokracovanie z last.pt a Adam.',
          'Povodny 4h experiment a jeho poradie ostavaju nezmenene.',
          'Rovnaky posledny LR, bez restartu, novych vstupov alebo zmeny architektury.',
          'Povodne a nove skutocne GPU-h sa scitavaju; nominalny strop linie je8h.',
          'Ziadne TEST ciele. Vsetky cisla nizsie su TRAIN/VAL diagnostika.','',
          'Model                  VAL4h       VAL8h-best   VAL8h-last   zlepsenie-best[%]   GPU-h-spolu']
    for r in models:
        text.append('%-23s %.7g  %.7g  %.7g  %12.3f  %12.3f'%(r['id'],r['baseline_VAL_loss'],r['continuation_best_VAL_loss'],r['continuation_last_VAL_loss'],r['phase_best_reduction_percent'],r['cumulative_GPU_h']))
    text+=['','Vybrany najlepsi priebezny model moze zostat povodny; posledny a novy',
           'fazovy najlepsi model sa vykazuju aj ked sa zhorsili. Cas prveho pozorovaneho',
           'zlepsenia je obmedzeny priblizne5min validacnym intervalom, nie presny minimalny cas.',
           'Krivky su v cost_curves.csv; povodny cosine rozvrh a novy konstantny LR',
           'sa rozlisuju. Z dvoch explorativnych seedov nerobime vseobecnu kauzalnu',
           'alebo statisticky vyznamnu vetu o celej rodine. Ziadne16h pokracovanie sa nespustilo.']
    (out/'VYSLEDKY.txt').write_text('\n'.join(text)+'\n')
    print('SEPARATE_SECOND_CHANCE_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
