# Porovnanie architektúr pre ustálené prúdenie

## Úplný prehľad dostupných dát

**[Katalóg výsledkov, váh a numerických polí](01_results/DOSTUPNE_DATA.md)** uvádza všetky publikované balíky, priame odkazy a aj výnimky z archivácie. Veľké súbory sú v Releases a nevyžadujú stiahnutie celého repozitára.

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

## Plné uložené predikčné polia

Všetkých 59 uložených TEST výstupov je dostupných v [samostatnom dátovom balíku](01_results/predikcne_polia/README.md).

## Historické numerické výstupy

Pôvodné polia a diagnostika starších pokusov sú v [samostatnom balíku](01_results/historicke_numericke_vystupy/README.md).

## Ďalšie výskumné záznamy

Doplnené pôvodné diagnostiky, protokoly, obrázky a zdroje sú v [samostatnom balíku](01_results/dalsie_vyskumne_zaznamy/README.md).

## Pôvodné Slurm logy

Pôvodné výstupné a chybové záznamy jednotlivých úloh Slurm sú v [samostatnom balíku](01_results/slurm_logy/README.md); zahŕňajú aj prípravné a neúspešné pokusy.

## Doplnenia z 19. novembra 2026

- [Výber architektúr, literatúra a hranice novosti](03_methodology/research_review_20261119/vyber_architektur_a_novost.txt)
- [Diagnostika ELM, SCN a CasCor](01_results/diagnostika_20261119/vysvetlenie_diagnostiky.txt)

Doplnenia zachovávajú pôvodné váhy a výsledky. Rozbory pracujú s pôvodnými TRAIN/VAL záznamami alebo so samostatnými diagnostickými kontrolami.

## Štatistické porovnanie architektúr — dokončené 24. 11. 2026

[Čitateľný výsledok, všetky intervaly a kontroly](01_results/statistical_comparison_20261123/README.md). Dodatočná analýza zahŕňa päť tréningov každej metódy, 200 TEST prípadov a 85 geometrických skupín, 200 000 krížených bootstrap replík a opravu na viacnásobné porovnávanie. TFNO má v tejto približnej analýze nižšie L2 chyby než všetkých osem hlavných alternatív; RealNPU je z nich najsilnejšie podľa oboch L2 metrík. Rozdiel TFNO–dense FNO nie je stabilne preukázaný. Obmedzenie iba piatich tréningov, posthoc charakter, samostatné kohorty aj nesplnenia 5 ms zostávajú uvedené. [Úplné resampling dáta a zdroje](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/architecture-statistics-20261123).
