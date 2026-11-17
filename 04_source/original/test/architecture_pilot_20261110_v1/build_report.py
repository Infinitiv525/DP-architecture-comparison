"""Generate a readable TXT report from the audited, small result records."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'report_152886'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    result = read(REPORT / 'summary.json')
    audit = read(REPORT / 'artifact_audit.json')
    assert result['successful'] == 10 and audit['status'] == 'PILOT_ARTIFACT_AUDIT_PASS'
    labels = dict(tfno='FNO/TFNO', chebkan='Chebyshev KAN', pinet='Pi-Net', dendrite='Dendrite Net',
                  realnpu='RealNPU', rbf='RBF NN', cascor='Cascade-Correlation', elm='ELM', scn='SCN',
                  pinn_reference='PINN (archivna)')
    rows = []
    for record in result['models']:
        name = record['model'];s = record['summary']
        cases = read(REPORT / name / 'case_metrics.json')
        rows.append(dict(model=labels[name], validation_mse=s['mean_validation_mse'],
            velocity_relative_l2_percent=100*s['mean_velocity_relative_l2'],
            pressure_relative_l2_percent=100*s['mean_pressure_relative_l2'],
            worst_case_gpu_p99_ms=s['worst_case_gpu_p99_ms'],
            worst_case_wall_p99_ms=max(c['timing']['synchronized_wall']['p99_ms'] for c in cases),
            new_training_seconds=s['training'].get('wall_seconds'),
            fitted_or_selected_core_scalars=read(REPORT / name / 'model_description.json')['data_fitted_or_selected_core_scalars']))
    with (REPORT / 'overview_extended.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text = '''VÝSLEDOK TECHNICKÉHO PILOTU ARCHITEKTÚR
Beh na Perune: 10.11.2026. Kontrola a uzavretie: 11.11.2026.

ČO SA PODARILO
Všetkých 10 vybraných položiek úspešne vytvorilo celé ustálené pole u,v,p
512 × 512 pre štyri validačné prípady. Deväť modelov bolo nanovo krátko
natrénovaných; desiaty je pôvodná PINN, ktorá sa iba načítala. Testovanie
neobsahuje časový vývoj simulácie. Ide o rovnakú ustálenú úlohu ako predtým.
Použili sme 16 TRAIN a 4 VAL prípady. TEST cieľové polia sa neotvárali.

Úloha 152886 má potvrdený stav COMPLETED a ExitCode 0:0. Bežala 94 sekúnd
na jednej H200 a štyroch CPU. Toto je krátky funkčný pilot, nie plný tréning.
Účtovanie sacct bolo nedostupné; 94 sekúnd je čas potvrdený riadiacou službou,
nie tvrdenie o konečnom zaúčtovaní portálom.

ORIENTAČNÉ VÝSLEDKY
MSE je priemer štvorcovej chyby celého poľa vrátane známeho telesa a troch
výstupov. L2 rýchlosti je spoločná relatívna chyba u,v vo fluide; L2 tlaku
sa ráta samostatne po odčítaní vlastného priemeru tlaku. Tabuľka uvádza
priemery štyroch prípadov, časy sú najhoršie p99 zo štyroch prípadov.
Vyše 100 % relatívnej chyby je možné: model je v tejto metrike horší než nula.

Model                 MSE        L2 rychl. %  L2 tlak %  GPU p99 ms  Wall p99 ms
'''
    for r in rows:
        text += f"{r['model']:<21} {r['validation_mse']:.7f}  {r['velocity_relative_l2_percent']:>10.2f}  {r['pressure_relative_l2_percent']:>9.2f}  {r['worst_case_gpu_p99_ms']:>10.3f}  {r['worst_case_wall_p99_ms']:>11.3f}\n"
    text += '''
ČO Z TOHO MÔŽEME POVEDAŤ
Všetkých 10 položiek v tejto skúške splnilo p99 do 5 ms pri výpočte celého
poľa. Na prípad bolo 100 GPU meraní a 30 synchronizovaných wall meraní;
na konečný záver o spoľahlivom p99 treba viac meraní a opakované behy.
Vstupy boli už na GPU. Čas zahŕňa opis geometrie závislý od vstupu, sieť,
masku a tlakovú referenciu. Prenos z PC a vykresľovanie sa nemerali.

Presnosť nových modelov zatiaľ nestačí. Najmä tlak má veľkú chybu. Po
120 Adam aktualizáciách nemožno rozhodnúť, čo dokážu po plnom tréningu.
Pri všetkých piatich bodových gradientových alternatívach klesla tréningová
chyba; RBF iba mierne. ELM, SCN a CasCor majú vlastné konštruktívne učenie.
Nehovoríme, že malá chyba archívnej PINN dokazuje lepšiu architektúru:
PINN už mala 640 TRAIN prípadov a 400 epoch, nové siete iba malý pilot.

Pred riadnym porovnaním treba uzavrieť vstupnú reprezentáciu. FNO používa
plné známe pole geometrie, bodové alternatívy jeho jednoduchý spoločný opis.
Rozdiel výsledku teda zatiaľ zahŕňa aj opis vstupu a tréningový postup.
Nové siete sú dátové modely; fyzikálne rovnice sa teraz používajú iba na
diagnostiku. Bez fyzikálnej tréningovej straty ich neoznačujeme za PINN.

ČO JE OVERENÉ
Pred behom prešlo 14 matematických kontrol a audit 20 použitých prípadov.
Po behu boli zo všetkých 40 uložených predikcií nanovo vypočítané metriky;
presne sa zhodujú s uloženými reportmi. Sedí aj p99 zo surových časov.
Všetkých deväť nových checkpointov sa obnovilo so striktnou kontrolou váh.
Pri starej PINN sedí pôvodný SHA256. Zdrojové kódy a freeze prostredia sú
overené hashmi. Osem bodových alternatív použilo tie isté tréningové body.

Pri obnove FNO treba použiť checkpoint_io.py: neuraloperator 2.0 vkladá
do state_dict vlastný slovník _metadata. Pomocný loader overí jeho zhodu
s konštruktorom a oddelí ho pred striktným načítaním obalu. Žiadna váha
sa neignoruje a pôvodný checkpoint ani zmrazené zdroje sa nemenili.

KDE SÚ VÝSLEDKY
Na Perune: /mnt/project/perun2601449/dp_llm_cfd/results/architecture_pilot_20261110_v1/job_152886/
Tam ostávajú váhy model.pt, optimizer_final.pt, plné prediction_*.npz,
raw_timings.npz, tréningové priebehy, metriky a obrázky comparison.png.
Starú PINN identifikuje pôvodná cesta a SHA256 v training/reference.
Lokálne v report_152886 sú iba malé metadáta, logy a 10 obrázkov; veľké
polia a váhy sa do PC nestiahli. Súhrn je v overview_extended.csv;
artifact_audit.json obsahuje kontroly a odtlačky vzdialených artefaktov.
Pôvodné kódy sú v revízii r_e9dafab653bac9e5 a jej archívnej kópii source/.

ČO NASLEDUJE
1. Ustáliť spoločný vstup pre osem alternatív a presne pomenovať, či
   porovnávame celé modely, alebo iba jadrá pri rovnakom opise geometrie.
2. Na TRAIN/VAL overiť dlhší tréning a rozumnú kapacitu každej siete;
   napríklad RBF inicializovať z tréningových bodov a neskúšať iba jednu
   náhodnú nevhodnú veľkosť. Pre všetky metódy určiť rovnaké pravidlá ladenia.
3. Pred záverečným porovnaním zmraziť konfigurácie, skórovanie a opakovania.
   Pôvodný TEST už bol v staršej štúdii otvorený; novú štúdiu neoznačiť
   za úplne nový slepý experiment. Veľký experiment sa týmto pilotom nespustil.
'''
    (HERE / 'vysledok_pilota.txt').write_text(text, encoding='utf-8')
    print('Readable report generated from audited results:', HERE / 'vysledok_pilota.txt')


if __name__ == '__main__':
    main()
