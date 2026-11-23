# Koľko ďalších tréningov a GPU hodín?

[Čitateľný výsledok a odporúčanie](VYSLEDKY.txt) · [Presné odporúčanie](recommendation.json) · [Všetkých 96 plánovacích scenárov](sample_size_design_summary.csv)

Odporúčaný cielený dodatok je **20 nových tréningov každej z metód TFNO, dense FNO a RealNPU**: 60 nových modelov, odhad 243 GPU-h vrátane TEST, plánovacia rezerva 260 GPU-h. Je navrhnutý na 90% silu pre približne 10% rozdiel chyby pri troch metrikách, s opravou na 9 porovnaní a konzervatívnym pilotným odhadom variability.

Rovnomerné nové porovnanie všetkých 10 architektúr vo všetkých dvojiciach potrebuje pri 90% sile pre 20% rozdiely dvoch L2 metrík **30 nových tréningov/metódu**: 300 modelov, odhad 977 GPU-h, tréningový strop a TEST rezerva 1312,5 GPU-h. MSE by v tomto novom protokole zostala sekundárna; pôvodná trojmetrická štúdia sa nemení.

Čísla sú podmienené plánovacími predpokladmi a pevným existujúcim benchmarkom. Nové tréningy nepridávajú geometrie a nemôžu samy odstrániť geometrickú neistotu. Výsledky nie sú univerzálnou hranicou obhájiteľnosti. **Žiadne nové tréningy sa nespustili.**

- [Nezávislá kontrola](independent_verification.json)
- [Náklady pôvodných 50 tréningov a TESTov](observed_training_and_TEST_costs.csv)
- [Geometrická neistota](geometry_uncertainty_limits.csv)
- [Podrobná metodika](../../03_methodology/power_analysis_20261124/protokol_v 2.txt)
- [Spustiteľný zdroj a nastavenie](../../04_source/power_analysis_20261124)
