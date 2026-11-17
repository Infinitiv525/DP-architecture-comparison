# Porovnanie architektúr pre ustálené prúdenie

Deväť rodín neurónových sietí dostalo rovnaké úplné geometrické vstupy, rovnaké dáta a najviac štyri hodiny tréningu na H200. Presnosť je výsledkom porovnania, nie vstupnou podmienkou.

Hlavný experiment má **45 sietí: 9 rodín × 5 seedov**. Samostatne je pridaných päť dense FNO, päť historických pomocných PINN a štyri predĺžené ChebKAN/RBF. Archívy obsahujú všetkých **59 vybraných modelov** aj dostupné ďalšie stavy. PINN referencia a dodatočné FNO nie sú zamieňané za pôvodný plne kontrolovaný deväťrodinový experiment.

[Porovnanie 55 sietí](01_results/final/POROVNANIE_55_MODELOV.txt) a [presné agregáty](01_results/final/summary_55.json). TFNO dosiahlo najnižšie priemerné L2 chyby; RealNPU bolo najsilnejšou z pôvodných alternatív podľa týchto chýb. Dense FNO bolo blízko TFNO a všetkých päť dense modelov splnilo p99 cieľ. Samostatný [TEST druhej šance](01_results/second_chance/VYSLEDKY.txt) zlepšil chybu všetkých štyroch predĺžených modelov, nemení však hlavné poradie pri štyroch hodinách.

FlowBench harmonics 2D, 512×512, 640 TRAIN / 160 VAL / 200 TEST. Predikujeme ustálené u/v/p, nie časový vývoj. Cieľ p99 ≤ 5 ms platí pre každý z 200 prípadov; nie je zárukou maximálneho času každého volania. Meranie začína so vstupmi už na GPU. Split zdieľa geometrie a TEST bol otvorený v starších štúdiách. Porovnanie priemerov samo nepreukazuje štatistickú významnosť. Autorský FlowBench kód používa TFNO; naša adaptácia nie je presnou reprodukciou jeho celého protokolu.

## Kde začať

| Priečinok | Čo v ňom nájdeš |
|---|---|
| [01_results](01_results) | Hotové porovnanie, grafy a číselné tabuľky |
| [02_models](02_models) | Index sietí a odkazy na skutočné váhy |
| [03_methodology](03_methodology) | Konfigurácie, metriky a podmienky experimentu |
| [04_source](04_source) | Vedecký zdrojový kód |
| [05_history](05_history) | Oddelená história pilotov a neúspešných pokusov |

## Modely a podrobné záznamy

[Stiahnuť modely a experimentálne záznamy](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/research-artifacts-v1). Archívy sú prílohy Release; bežný `git clone` ich nesťahuje. [Index modelov](02_models/model_index.csv) určuje príslušný diel archívu, pôvod a SHA256 každého súboru. [Manifest](02_models/ARTIFACT_MANIFEST.json) pokrýva aj zverejnené experimentálne záznamy.

Zachovaných je **587 checkpointových súborov**, vrátane vybraných váh, dostupných posledných stavov a obnovovacích stavov. Toto číslo nie je počtom nezávislých tréningov. Finálne siete a história sa v archíve aj indexe rozlišujú. Vývojové a neúspešné pokusy nenahrádzajú finálne výsledky.

Úplné veľké vstupné FlowBench/CFD polia a stiahnuté pôvodné LLM sa neduplikujú; ich zdroje a pripnuté verzie sú v metodike. Surové natívne predikčné polia možno znovu vypočítať zo zachovaných sietí. Kód obsahuje pôvodné pracovné cesty a vyžaduje nastavenie prostredia pre daný stroj. Archívy zachovávajú váhy byte za bytom.

## Súvisiace experimenty

- [3D experimenty](https://github.com/Infinitiv525/DP-3D-experiments)
- [Porovnanie LLM](https://github.com/Infinitiv525/DP-LLM-comparison)
- [Porovnanie architektúr](https://github.com/Infinitiv525/DP-architecture-comparison)

Verejný vedecký export oddeľuje tieto podklady od súkromnej rozpracovanej dizertácie a prevádzkovej histórie. Referenčné dáta pochádzajú z [FlowBench](https://huggingface.co/datasets/BGLab/FlowBench); metódy autorov opisuje [pôvodný projekt](https://github.com/baskargroup/flowbench-tools). Licencie a zdroje externých vstupov sú v [metodike](03_methodology/DATA_AND_REUSE.md).

## Doplnkové vedecké dáta

Ďalšie surové merania a podklady na reprodukciu sú v [doplnkovom balíku](01_results/doplnkove_data/README.md).
