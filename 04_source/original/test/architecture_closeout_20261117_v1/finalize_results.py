"""Close the verified reports; never edit frozen weights, metrics or sources."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
TEST=HERE.parent
OLD=TEST/'architecture_budget_test_20261112_v1'
FNO=TEST/'architecture_dense_fno_20261112_v1'
SECOND=TEST/'architecture_second_chance_20261112_v1'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    text=value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False)+'\n'
    text='\n'.join(line.rstrip() for line in text.splitlines())+'\n'
    Path(path).write_text(text,encoding='utf-8',newline='\n')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    proofs=[(FNO,'remote_inventory','report'),(SECOND,'inventory','report/continuation')]
    checked=0
    for folder,key,subfolder in proofs:
        for name,sha in read(folder/'completion_check.json')[key]['small_report_sha256'].items():
            assert digest(folder/subfolder/name)==sha,name
            checked+=1
    original=read(OLD/'report/audit/summary.json')
    dense=read(FNO/'report/audit/summary.json')
    second=read(SECOND/'report/continuation/summary/complete.json')
    assert original['status']=='FINAL50_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
    assert dense['status']=='DENSE_FNO5_TEST_AND_SAVED_OUTPUT_AUDIT_COMPLETE'
    assert second['status']=='SEPARATE_SECOND_CHANCE_COMPLETE'
    assert second['excluded_from_primary_experiment'] and second['no_TEST_targets']
    models=original['models']+dense['models'];assert len(models)==len({r['id'] for r in models})==55
    methods=original['methods']+dense['methods'];assert len(methods)==11
    order=['tfno','fno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor','pinn']
    methods=sorted(methods,key=lambda r:order.index(r['model']))
    for m in methods:
        rows=[r for r in models if r['model']==m['model']];assert len(rows)==5
        for key in ['all_cell_mse','fluid_velocity_relative_l2','fluid_pressure_relative_l2']:
            assert abs(statistics.mean(r[key] for r in rows)-m[key+'_mean'])<1e-12
    summary=dict(status='VERIFIED_55_MODEL_TEST_COMPARISON_CLOSED',
        reported_on='2026-11-17',raw_results_verified_on='2026-11-13',
        original_primary_models=45,posthoc_dense_FNO_models=5,auxiliary_PINN_models=5,
        total_TEST_models=55,TEST_cases_per_model=200,total_saved_TEST_fields=11000,
        raw_timing_pairs=11000000,methods=methods,models=models,
        separate_exploratory_lineages=4,exploratory_results_not_merged=True,
        known_geometry_split=True,previous_TEST_exposure_disclosed=True,
        no_statistical_significance_claim=True,
        sources=dict(original=str(OLD/'report/audit/summary.json'),dense=str(FNO/'report/audit/summary.json'),
                     separate_exploratory=str(SECOND/'report/continuation/summary/complete.json')))
    write(HERE/'summary_55.json',summary)
    text=['UZAVRETÉ POROVNANIE ARCHITEKTÚR — STAV K 17. 11. 2026','',
        'Hotové je pôvodné porovnanie deviatich rodín po päť modelov, päť',
        'dodatočných dense FNO modelov a päť pomocných PINN referencií.',
        'Spolu 55 uložených modelov × 200 TEST scenárov = 11 000 predikčných polí.',
        'Surové časovania obsahujú 11 miliónov párov GPU/host meraní.',
        'Druhá šanca ChebKAN/RBF ostáva v samostatnom reporte a do tabuľky sa nepridáva.','',
        'Metóda       Rýchlosť L2 [%]       Tlak L2 [%]        Splnené p99≤5ms',
        '------------------------------------------------------------------------']
    labels={'tfno':'TFNO','fno':'FNO*','chebkan':'ChebKAN','pinet':'PiNet','dendrite':'Dendrite',
            'realnpu':'RealNPU','rbf':'RBF','elm':'ELM','scn':'SCN','cascor':'CasCor','pinn':'PINN**'}
    for r in methods:
        text.append('%-11s %7.3f ± %-6.3f   %8.3f ± %-6.3f      %d/5'%(labels[r['model']],
            100*r['fluid_velocity_relative_l2_mean'],100*r['fluid_velocity_relative_l2_sample_sd'],
            100*r['fluid_pressure_relative_l2_mean'],100*r['fluid_pressure_relative_l2_sample_sd'],r['host_5ms_pass_models']))
    text+=['','Chyby sú priemery relatívnych L2 chýb v tekutine. ± je výberová SD',
        'piatich modelových priemerov; každý modelový priemer obsahuje 200 prípadov.',
        'P99 limit splní model iba vtedy, keď p99 neprekročí 5 ms v žiadnom',
        'z jeho 200 TEST prípadov. Neznamená to maximálny čas každého volania.',
        'Meria sa celý výstup ustáleného poľa u/v/p so vstupom už na H200,',
        'batch 1, FP32, TF32 vypnuté. Prenos dát a vykreslenie sú mimo merania.',
        'Nie je to meranie časového kroku dynamickej simulácie.','',
        '* Dense FNO je dodatočný variant, pridaný až po znalosti pôvodných TEST',
        'výsledkov. Má rovnaké vstupy, stratu a najviac 4h rozpočet, ale odstránená',
        'faktorizácia zvýšila počet parametrov. Pôvodní autori FlowBench vo svojom',
        'kóde sami používajú TFNO; naša adaptácia nie je presná reprodukcia článku.',
        '** PINN je historická pomocná referencia s inou reprezentáciou a učením;',
        'nie je plne kontrolovaným súťažiacim nového deväťrodinového protokolu.',
        'Split obsahuje známe geometrie a TEST už bol otvorený v starších štúdiách.',
        'Tabuľka je opisné porovnanie, nevyhlasuje štatistickú významnosť.','',
        'Čo sa dá z výsledkov priamo povedať',
        'TFNO má najnižšie priemerné relatívne chyby rýchlosti a tlaku. Dense FNO',
        'je v týchto chybách blízko a všetkých päť jeho modelov splnilo p99 cieľ.',
        'RealNPU je z ôsmich pôvodných alternatív najsilnejšie podľa priemerných',
        'relatívnych chýb a všetkých päť modelov splnilo aj časový cieľ.',
        'Poradie sa môže líšiť podľa použitej metriky; úplné MSE aj fyzikálne',
        'diagnostiky sú v pôvodných prípadových reportoch, nie iba v tejto tabuľke.','',
        'Samostatná druhá šanca',
        'ChebKAN a RBF pokračovali s dvoma vopred určenými seedmi ďalšie štyri',
        'GPU hodiny pri zachovanom Adam a koncovom LR. ChebKAN znížilo VAL stratu',
        'o 11,29 % a 15,21 %, RBF o 8,13 % a 4,99 %. Dodatok spotreboval podľa',
        'closeoutov 15,6678 dodatočných H200-h. Celý čas jednej línie bol asi',
        '7,834 h pri nominálnom strope 8 h. Toto nie sú nové TEST výsledky.',
        'Dodatok ukazuje mierny prínos ďalšieho učenia pri konkrétnom koncovom LR,',
        'nepreukazuje minimálny potrebný čas ani optimálne nastavenie architektúry.','',
        'Kde sú podrobnosti',
        '../architecture_dense_fno_20261112_v1/VYSLEDKY.txt — dense FNO.',
        '../architecture_second_chance_20261112_v1/VYSLEDKY.txt — druhá šanca.',
        '../architecture_training_diagnostics_20261112_v1/VYSVETLENIE.txt — diagnostika 45 modelov.',
        'summary_55.json — presné agregáty a všetkých 55 modelových súhrnov.',
        'Váhy, natívne polia a surové časovacie polia ostávajú na Perune.',
        'Žiadny ďalší tréning ani šestnásťhodinové pokračovanie sa teraz nespúšťa.']
    write(HERE/'POROVNANIE_55_MODELOV.txt','\n'.join(text)+'\n')

    m=dense['methods'][0];proof=read(FNO/'completion_check.json')
    fno_models=dense['models'];p99=[r['worst_host_p99_ms'] for r in fno_models]
    fno_text=['DENSE FNO — TRÉNINGY, TESTOVANIE AJ AUDIT SÚ DOKONČENÉ','',
        'Päť nezávislých tréningov na 640 TRAIN / 160 VAL prípadoch.',
        'Každý vybraný model bol vyhodnotený na rovnakých 200 TEST prípadoch.',
        'Spolu 1 000 uložených natívnych polí a milión párov GPU/host časovaní.',
        'CPU audit overil všetky uložené polia, fyzikálne diagnostiky a časovania.','',
        'Priemerná relatívna L2 chyba rýchlosti: %.3f %% ± %.3f percentuálneho bodu.'%(100*m['fluid_velocity_relative_l2_mean'],100*m['fluid_velocity_relative_l2_sample_sd']),
        'Priemerná relatívna L2 chyba tlaku: %.3f %% ± %.3f percentuálneho bodu.'%(100*m['fluid_pressure_relative_l2_mean'],100*m['fluid_pressure_relative_l2_sample_sd']),
        'Priemerná natívna MSE u/v/p: %.9g.'%m['all_cell_mse_mean'],
        'Časový cieľ najhoršieho scenárového p99 ≤ 5 ms splnilo všetkých 5/5 modelov.',
        'Rozsah najhorších host p99 medzi modelmi: %.3f až %.3f ms.'%(min(p99),max(p99)),
        'P99 nie je maximálny čas každého volania. H2D prenos a vizualizácia',
        'sú mimo merania; vstup je na H200, FP32, batch 1, TF32 vypnuté.','',
        'Ide o FNO bez faktorizácie, s rovnakým backbone a plnými vstupmi ako náš',
        'TFNO: päť vrstiev, 48 skrytých kanálov a módy 32 × 32.',
        'Má 12 672 888 trénovateľných reálnych skalárov plus 10 normalizačných hodnôt.',
        'Pridelený rozpočet na model bol rovnaký: najviac 4 H200-h a 16 CPU-core-h.',
        'Päť finálnych tréningov podľa uložených closeoutov spotrebovalo spolu',
        '%.4f H200-h. Ladenie a záverečné meranie sú samostatné položky.'%proof['training_H200_hours'],
        'Closeout runtime nie je presné portálové vyúčtovanie.','',
        'Autori FlowBench v zverejnenom modeli sami volajú TFNO. Táto dense verzia',
        'je dodatočný variant; nie je chýbajúcim presným pôvodným baseline.',
        'Bola pridaná po znalosti pôvodných výsledkov. TEST slúžil na vyhodnotenie,',
        'nie na výber jej checkpointov alebo zmenu konfigurácie.',
        'Pôvodný deväťrodinový experiment a pomocná PINN zostali zachované.','',
        'Úplné čísla: report/audit/summary.json, case_metrics.csv a model_metrics.csv.',
        'Presné váhy a SHA: report/frozen_evaluation/models.json.',
        'Integrita: completion_check.json — päť úspešných tréningových closeoutov,',
        'desať VAL kontrolných polí a tisíc CPU overených TEST polí.',
        'Staré finálne Slurm záznamy už expirovali; netvrdíme ich spätné overenie.',
        'Pôvodné reporty a chybové logy sú kompletné, stderr prázdne.',
        'Váhy a objemné polia ostávajú na Perune.','',
        'Kompletné porovnanie je v ../architecture_closeout_20261117_v1/POROVNANIE_55_MODELOV.txt.']
    write(FNO/'VYSLEDKY.txt','\n'.join(fno_text)+'\n')

    second_text=['DRUHÁ ŠANCA CHEBKAN A RBF — DOKONČENÁ A OVERENÁ','',
        'Samostatný exploratívny doplnok mimo hlavného experimentu.',
        'Dve vopred určené pokračovania každej rodiny, zo skutočného last.pt',
        'vrátane Adam momentov a presného poradia prípadov. Bez zmeny architektúry.',
        'Posledný uložený LR približne 1e-5 zostal konštantný, bez reštartu.',
        'Celkový nominálny rozpočet bol 4 + 4 = 8 GPU-h na líniu.','',
        'Model a seed          VAL pred     VAL nový best   VAL posledný   Zníženie [%]',
        '--------------------------------------------------------------------------------']
    for r in second['models']:
        second_text.append('%-21s %.8f   %.8f     %.8f       %6.2f'%(r['architecture']+' '+str(r['seed']),r['baseline_VAL_loss'],r['continuation_best_VAL_loss'],r['continuation_last_VAL_loss'],r['phase_best_reduction_percent']))
    second_text+=['','Ide o spoločnú natívnu vyváženú VAL stratu, nie percentuálnu L2 chybu.',
        'Vo všetkých štyroch líniách sa zlepšil nový best aj posledný stav.',
        'Najlepší priebežný a posledný model uvádzame osobitne, aby bolo vidno',
        'aj zhoršenie po predchádzajúcom lepšom checkpointe.','',
        'Relatívne L2 chyby na 160 VAL prípadoch, pôvodný → vybraný nový model:',
        'Model a seed          Rýchlosť [%]        Tlak [%]']
    for r in second['models']:
        second_text.append('%-21s %6.3f → %-6.3f    %6.3f → %-6.3f'%(r['architecture']+' '+str(r['seed']),100*r['baseline_velocity_L2'],100*r['selected_velocity_L2'],100*r['baseline_pressure_L2'],100*r['selected_pressure_L2']))
    second_text+=['','Cena a interpretácia',
        'Dodatočné H200-h podľa closeoutov spolu: %.4f.'%second['total_additional_observed_GPU_hours'],
        'Dodatočné pridelené CPU-core-h tréningov: %.4f.'%(4*second['total_additional_observed_GPU_hours']),
        'Každé pokračovanie spotrebovalo asi 3,917 H200-h. S pôvodným behom',
        'to bolo asi 7,834 H200-h na líniu. Technické CPU kontroly sú oddelená réžia.',
        'Runtime pri closeoute zahŕňa startup a validácie, nie iba gradientové kroky;',
        'nie je presným portálovým vyúčtovaním.',
        'Zlepšenie existuje, ale prírastok je mierny oproti približne dvojnásobným',
        'zdrojom. Ukazuje citlivosť tejto konkrétnej konfigurácie pri koncovom LR,',
        'nie minimálny potrebný čas celej rodiny alebo jej najlepšie možné nastavenie.',
        'Dva seedy neposkytujú rovnaký rozsah dôkazov ako hlavná päťseedová séria.',
        'Nepočítali sa nové TEST výsledky. Tieto VAL čísla sa nezmiešajú s TEST',
        'tabuľkou pôvodných modelov. Hlavné poradie, výsledky a váhy sa nemenia.','',
        'Číselná kontrola',
        'Štyri úspešné tréningové closeouty a štyri CPU audity. Pri pôvodnom best,',
        'novom best a novom last sa prepočítalo spolu 9 600 TRAIN/VAL prípadových',
        'strát a natívne VAL chyby rýchlosti a tlaku. SHA rodičov ostali nezmenené.',
        'Pri RBF sa pôvodný float32 Adam step nasýtil na 16 777 216; skutočné',
        'počty krokov sú 18 061 736 a 17 968 042, potvrdené celočíselnými počítadlami.',
        'Stav a dtype Adam sa zámerne nemenili. V overenom kóde ovplyvňuje step',
        'bias korekcie 1-beta^t; pri nasýtenom aj skutočnom t sú obe beta^t',
        'numericky nula, takže tieto korekcie zostávajú rovné jednej. Zdrojový',
        'hash a číselný výpočet sú v adam_counter_check.json. Nezamieňame preto',
        'interný float32 čítač za skutočný počet vykonaných aktualizácií.','',
        'Kde sú podklady',
        'report/continuation/summary/model_metrics.csv — všetky štyri línie.',
        'report/continuation/summary/cost_curves.csv — pôvodné a nové krivky.',
        'report/continuation/audit/<id>/complete.json — prípadové chyby a fyzikálne jednotky.',
        'completion_check.json — súhrn kontrol a zhoda vzdialených reportov.',
        'Krivka je v figures/druha_sanca_priebeh.png a .svg.',
        'Checkpointy ostávajú na Perune. Šestnásťhodinové pokračovanie nie je spustené.']
    write(SECOND/'VYSLEDKY.txt','\n'.join(second_text)+'\n')

    for folder,status,body in [
        (FNO,'DENSE_FNO5_TRAIN_TEST_AND_CPU_AUDIT_VERIFIED',
         'Päť tréningov, päť TEST vyhodnotení po 200 prípadoch a CPU audit sú hotové.\nRýchlosť L2 0,811 %, tlak L2 1,666 %, p99 cieľ splnilo 5/5 modelov.\n'),
        (SECOND,'SECOND_CHANCE_FOUR_LINEAGES_COMPLETE_AND_AUDITED',
         'Štyri oddelené pokračovania aj CPU audity sú hotové.\nChebKAN: VAL strata nižšia o 11,29 % a 15,21 %. RBF: o 8,13 % a 4,99 %.\nDodatočné H200-h spolu 15,6678. Samostatný doplnok mimo hlavného experimentu.\n')]:
        state=read(folder/'current_status.json');state['status']=status;state['written_closeout_date']='2026-11-17'
        if folder==FNO:
            state.update(main_models_complete=5,main_models_running=0,test_models_complete=5,CPU_audit_complete=True,saved_TEST_fields=1000)
        else:
            state.update(continuation_models_complete=4,continuation_models_running=0,CPU_audits_complete=4,
                         additional_observed_GPU_hours=second['total_additional_observed_GPU_hours'],no16h_extension_scheduled=True)
        write(folder/'current_status.json',state)
        write(folder/'STAV.txt','VŠETKO JE DOKONČENÉ A OVERENÉ\n\n'+body+'\nČitateľný výsledok: VYSLEDKY.txt. Dôkazy: completion_check.json.\nVýpočty overené 13. 11.; písomné uzavretie aktualizované 17. 11. 2026.\nŽiadne nové tréningy sa pri tomto uzavretí nespúšťajú.\n')
        protocol=(folder/'protokol.txt').read_text(encoding='utf-8')
        marker='\nUZAVRETIE 17. 11. 2026\n'
        if marker not in protocol:
            write(folder/'protokol.txt',protocol+marker+body+'\nPôvodný postup vyššie je záznam spustenia. Aktuálny stav je dokončený;\nčíselné reporty a zmrazené zdroje sa týmto uzavretím nemenia.\n')
    diag=TEST/'architecture_training_diagnostics_20261112_v1'
    state=read(diag/'current_status.json');state.update(exploratory_second_chance_queued=False,
         exploratory_second_chance_complete=True,second_chance_kept_outside_main_experiment=True)
    write(diag/'current_status.json',state)
    for name in ['STAV.txt','VYSVETLENIE.txt','druha_sanca_navrh.txt']:
        value=(diag/name).read_text(encoding='utf-8');marker='\nUZAVRETIE DRUHEJ ŠANCE 17. 11. 2026\n'
        if marker not in value:
            write(diag/name,value+marker+'Štyri autorizované pokračovania sú dokončené a skontrolované. ChebKAN\nznížilo VAL stratu o 11–15 %, RBF o 5–8 %, pri približne dvojnásobnom\nrozpočte línie. Ide o samostatný doplnok; hlavné výsledky sa nemenia.\nÚplný výstup: ../architecture_second_chance_20261112_v1/VYSLEDKY.txt.\n')
    receipt=dict(status='RESULTS_WRITTEN_CLOSEOUT_COMPLETE',date='2026-11-17',
        verified_new_report_hashes=checked,TEST_models=55,TEST_fields=11000,
        independent_original_primary_runs=45,posthoc_dense_FNO_runs=5,auxiliary_PINN_references=5,
        separate_exploratory_lineages=4,exploratory_results_excluded_from_primary=True,
        no_new_jobs_submitted=True,large_remote_artifacts_not_downloaded=True,
        source_proofs={str(p.relative_to(TEST)):digest(p) for p in [OLD/'completion_check.json',FNO/'completion_check.json',SECOND/'completion_check.json']},
        outputs={str(p.relative_to(TEST)):digest(p) for p in [HERE/'summary_55.json',HERE/'POROVNANIE_55_MODELOV.txt',FNO/'VYSLEDKY.txt',SECOND/'VYSLEDKY.txt']})
    for p in [SECOND/'figures/druha_sanca_priebeh.png',SECOND/'figures/druha_sanca_priebeh.svg',SECOND/'plotting_environment.json']:
        if p.exists():receipt['outputs'][str(p.relative_to(TEST))]=digest(p)
    previous=read(HERE/'closeout_receipt.json') if (HERE/'closeout_receipt.json').exists() else {}
    for key in ['isolated_plotting_dependencies_bytes','temporary_plotting_cleanup_blocked_by_automatic_policy','plotting_dependencies_excluded_from_git']:
        if key in previous:receipt[key]=previous[key]
    write(HERE/'closeout_receipt.json',receipt)
    print('Closed: 55 TEST models, 11000 fields; separate four-lineage second chance.')
    print('Matched archived report hashes:',checked)


if __name__=='__main__':main()
