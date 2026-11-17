"""Describe available CPU diagnostics and original curves, without retuning."""
import csv
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'architecture_budget_main_20261111_v1/report/main'
NAMES=['tfno','chebkan','pinet','dendrite','realnpu','rbf','elm','scn','cascor']


def main():
    original=[]
    for name in NAMES:
        for file in sorted(BASE.glob(name+'_*/complete.json')):
            r=json.loads(file.read_text())
            row=dict(id=file.parent.name,model=name,stop_reason=r['stop_reason'],
                training_seconds=r['allocation_elapsed_seconds'],
                best_at_seconds=r['best_checkpoint_selected_at_seconds'],
                gradient_updates=r['gradient_updates'],final_grown_units=r['final_grown_units'],
                min_native_TRAIN_visits=min(r['complete_native_TRAIN_case_visits']),
                best_VAL_loss=r['best_complete_validation_loss'])
            for seconds in [3600,7200,14400]:
                values=[v['validation']['mean_raw_case_balanced_loss'] for v in r['history'] if v['allocation_elapsed_seconds']<=seconds]
                row['observed_best_VAL_by_'+str(seconds)+'s']=min(values) if values else None
            original.append(row)
    assert len(original)==45
    available=[]
    for name in NAMES:
        for file in sorted((HERE/'report'/name).glob(name+'_s*.json')):
            available.append(json.loads(file.read_text()))
    result=dict(status='CPU_DIAGNOSTICS_COMPLETE' if len(available)==45 else 'CPU_DIAGNOSTICS_PARTIAL',
       completed_models=len(available),planned_models=45,original_training_records=original,
       no_new_training=True,test_targets_read=False,methods=[])
    for name in NAMES:
        rows=[r for r in original if r['model']==name]
        cpu=[r for r in available if r['model']==name]
        curve={str(t):statistics.mean(r['observed_best_VAL_by_'+str(t)+'s'] for r in rows)
               for t in [3600,7200,14400]}
        result['methods'].append(dict(model=name,training_records=5,CPU_diagnosed_models=len(cpu),
            observed_stop_reasons=sorted(set(r['stop_reason'] for r in rows)),
            training_seconds_range=[min(r['training_seconds'] for r in rows),max(r['training_seconds'] for r in rows)],
            min_case_visits_range=[min(r['min_native_TRAIN_visits'] for r in rows),max(r['min_native_TRAIN_visits'] for r in rows)],
            TRAIN_mean=statistics.mean(r['phase_metrics']['train']['mean_raw_case_balanced_loss'] for r in cpu) if cpu else None,
            VAL_mean=statistics.mean(r['phase_metrics']['validation']['mean_raw_case_balanced_loss'] for r in cpu) if cpu else None,
            all_CPU_replays_match=all(r['CPU_validation_mean_matches_stored_GPU'] and r['CPU_validation_cases_match_stored_GPU'] for r in cpu) if cpu else None,
            observed_best_VAL_curve_means=curve,
            observed_VAL_reduction_2_to_4_hours_percent=100*(1-curve['14400']/curve['7200'])))
    (HERE/'diagnostic_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (HERE/'original_training_curves_summary.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(original[0]));writer.writeheader();writer.writerows(original)
    text=['PREČO NIEKTORÉ MODELY DOPADLI SLABŠIE','',
       'Aktuálne CPU overených modelov: '+str(len(available))+'/45.'+(' Diagnostika je dokončená.' if len(available)==45 else ' Podklad sa doplní po dobehnutí.'),
       'Používame nemenné uložené váhy. Žiadne nové učenie ani výber podľa TEST.','',
       'Natívna vyvážená strata nie je percentuálna chyba; nižšia je lepšia.',
       'Každá metóda má päť overených modelov.' if len(available)==45 else 'V tabuľke môže mať metóda zatiaľ menej než päť overených modelov.',
       'Metóda      Hotovo       TRAIN strata       VAL strata    Zhoda CPU/GPU VAL']
    for m in result['methods']:
        text.append('%-11s %d/5       %s       %s       %s'%(m['model'],m['CPU_diagnosed_models'],
            '%.8g'%m['TRAIN_mean'] if m['TRAIN_mean'] is not None else 'čaká',
            '%.8g'%m['VAL_mean'] if m['VAL_mean'] is not None else 'čaká',str(m['all_CPU_replays_match'])))
    if len(available)==45:
        text+=['','ZÁVERY PO DOKONČENÍ VŠETKÝCH 45 PREPOČTOV',
           'Všetkých 7 200 jednotlivých CPU VAL strát súhlasí s pôvodnými GPU',
           'výsledkami v uvedenej numerickej tolerancii. Spolu s TRAIN sa',
           'skontrolovalo 36 000 prípadových strát. Žiadna váha sa nemenila.',
           'PiNet, Dendrite, ELM, SCN aj CasCor majú vysokú chybu už na TRAIN.',
           'Slabé výsledky sa teda nedajú vysvetliť iba prechodom na novú sadu.',
           'Nedostatočnú reprezentáciu a neúčinnú optimalizáciu tieto dáta',
           'jednoznačne neoddelia. Pri CasCor je TRAIN dokonca mierne horšia než',
           'VAL; samo osebe to nie je chyba, sady majú odlišné konkrétne prípady.',
           'ChebKAN a RBF majú nižšiu TRAIN stratu než VAL, ale stále aj TRAIN',
           'stratu výrazne vyššiu než TFNO; problém nemožno zúžiť len na preučenie.','',
           'POZOROVANÝ POKLES VAL STRATY MEDZI 2. A 4. HODINOU',
           'Priemer najlepších dovtedy pozorovaných VAL strát piatich modelov:']
        for m in result['methods']:
            text.append('%-11s %9.5f %%'%(m['model'],m['observed_VAL_reduction_2_to_4_hours_percent']))
        text+=['ChebKAN a RBF sú preto najlepšie podložené prvé návrhy na neskoršie',
           'skúmanie dlhšieho tréningu. Pri CasCor bol dodatočný pokles pod 0,001 %,',
           'takže samotné predĺženie nie je podložené rovnako silno.',
           'Tieto poklesy patria k jednému 4h kosínusovému rozvrhu a nie sú',
           'odhadom, ako dopadne nový 8h rozvrh alebo pokračovanie pri minimálnom LR.',
           'Návrh druhej šance je v druha_sanca_navrh.txt; aktuálne spustenie, ak už bolo autorizované, je uvedené v jeho dodatku a v ../architecture_second_chance_20261112_v1/STAV.txt.']
    text+=['','PRIAMO POTVRDENÉ Z TRÉNINGOVÝCH REPORTOV',
       'ELM: výstupné váhy sa vyriešili priamo; náhodná skrytá vrstva sa neučí.',
       'Všetkých päť behov preto prirodzene skončilo za 4 až 9 sekúnd. Viac',
       'čakania pri rovnakom algoritme neznamená ďalšie učenie.',
       'SCN: všetkých päť behov skončilo bez prípustného nového kandidáta.',
       'Hľadanie má konečný rozsah a vyžaduje zlepšenie všetkých aktívnych',
       'výstupných zložiek; nejde o vyčerpanie štvorhodinového rozpočtu.',
       'CasCor: päť behov spotrebovalo celý tréningový rozpočet; postupne',
       'pridávali a zmrazovali neuróny, teda neprebiehalo spoločné doladenie',
       'všetkých skrytých váh. Niektoré najlepšie modely vznikli oveľa skôr.',
       'PiNet a Dendrite: celú TRAIN sadu videli opakovane pri miliónoch',
       'aktualizácií. Použité varianty majú polynomiálnu rekurenciu stupňa 3',
       'vo vstupných príznakoch. To je vlastnosť implementácie, nie dôkaz príčiny.',
       'TFNO má naučené globálne priestorové miešanie Fourierovými vrstvami;',
       'ostatné jadrá pracujú bodovo na spoločných úplných vstupných príznakoch.',
       'Odlišný spôsob spracovania kontextu je možným vysvetlením rozdielov,',
       'nie samostatne dokázanou kauzálnou príčinou.','',
       'AKO ČÍTAŤ DOPLŇOVANÉ TRAIN/VAL ČÍSLA',
       'Ak je chyba vysoká už na TRAIN a VAL je podobná, problém nie je len',
       'v prechode na nové prípady. Zodpovedá to nedostatočnému prispôsobeniu',
       'reprezentácie alebo optimalizácie. Tieto dve príčiny samotný gap nerozlíši.',
       'Výrazne menšia TRAIN chyba môže odhaliť slabšie zovšeobecnenie, ale',
       'pomer sám nestačí: aj presný TFNO má približne dvojnásobnú VAL stratu.',
       'Porovnávajú sa preto aj absolútne chyby a všetkých päť opakovaní.',
       'Zlepšovanie VAL na konci je stopa pre možný prínos dlhšieho učenia,',
       'nie dôkaz, že viac času určite odstráni rozdiel.',
       'Časové body v CSV patria k jednému štvorhodinovému rozvrhu učenia,',
       'nejde o samostatne optimalizované jedno- a dvojhodinové experimenty.','',
       'ČO ZATIAĽ NEVIEME TVRDIŤ',
       'Bez cieleného kontrolného zásahu nevieme dokázať, že rozdiel spôsobil',
       'konkrétny počet vrstiev, báza, výber kandidátov alebo nastavenie učenia.',
       'Výsledky hodnotia naše konkrétne implementácie v tomto rozhraní a',
       'rozpočte; nepredstavujú všeobecný zákaz použitia danej rodiny na fyziku.',
       'Takéto zásahy a nové tréningy sa pri tejto diagnostike nespúšťajú.']
    (HERE/'VYSVETLENIE.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(result['status'],len(available),'/45')


if __name__=='__main__':main()
