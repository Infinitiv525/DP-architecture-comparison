# Štatistické porovnanie architektúr

**[Výsledky ľudskou rečou](VYSLEDKY.txt)** · [Všetky kontrasty](pairwise_comparisons.csv) · [Súhrn metód](method_summary.csv)

Analýza zohľadňuje päť tréningov každej architektúry a 85 geometrických skupín v 200 TEST prípadoch. Používa 200 000 krížených bootstrap replík, Bonferroni intervaly a Holm upravené približné p-hodnoty. Je dodatočná a exploratívna; iba päť tréningov obmedzuje istotu záverov.

V rámci tejto analýzy TFNO dosiahlo nižšie relatívne L2 chyby rýchlosti aj tlaku než všetkých osem hlavných alternatív. RealNPU bolo najsilnejšou z ôsmich alternatív podľa oboch L2 metrík. Rozdiel TFNO oproti dodatočnému dense FNO nepovažujeme za stabilne preukázaný. Pôvodné váhy, metriky a merania ostali zachované.

- [Opatrná klasifikácia všetkých kontrastov](interpretation.csv)
- [Úplné presné intervaly, p-hodnoty a kontroly](contrasts.json)
- [Obmedzenie piatich seedov — znamienková kontrola](exact_seed_sensitivity.csv)
- [Druhá šanca — len dva konkrétne tréningové seedy](second_chance_conditional.csv)
- [Latencia — opisné vyhodnotenie](latency_descriptive.csv)
- [Nezávislé overenie výpočtov](independent_verification.json)
- [Vstupné SHA a nemenný pôvod údajov](input_manifest.json)
- [Podrobná metodika](../../03_methodology/statistical_comparison_20261123/protokol.txt)
- [Spustiteľný kód a nastavenie](../../04_source/statistical_comparison_20261123)
- [Všetky resampling dáta a zdroje v Release](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/architecture-statistics-20261123)

Ani nominálny bootstrap interval, ani nezamietnutie nulovej hypotézy nie sú dôkazom ekvivalencie alebo univerzálnej nadradenosti rodiny sietí. Rozsahom sú tieto implementácie, známe geometrie FlowBench a konkrétny tréningový rozpočet.
