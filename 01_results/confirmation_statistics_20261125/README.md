# Potvrdenie troch architektúr na20nových tréningoch
[Čitateľný výsledok](VYSLEDKY.txt) · [Všetky primárne testy](paired_t_primary.json) · [Nezávislá kontrola](independent_verification.json).
TFNO má na pevnom200-prípadovom benchmarku nižšie chyby než denseFNO ajRealNPU vo všetkých3metrikách. Osem z9vopred určených párových kontrastov prešloHolm korekciou; celkováMSE denseFNO–RealNPU sa nepreukázala.
20nových semien/metódu; pôvodných5sa nemieša. [Priemery jednotlivých modelov](per_seed_model_means.csv).
**Obmedzenie:**opakovateľnosť cez tréningové semená na známom benchmarku. Pri približnom geometrickom bootstrape sú záveryTFNO–FNO citlivé na druh intervalu; všeobecnú prevahu na nevidených geometriách netvrdíme.
Všetkých60modelov splnilo per-casep99<=5ms naH200. Jednotlivé prekročenia sú priznané v reporte.
[Protokol](../../03_methodology/confirmation_statistics_20261125/protokol.txt) · [Zdroj](../../04_source/confirmation_statistics_20261125).
Úplné váhy, polia, numerické replikáty a zdroje:odkazy pribudnú vARCHIV.txt po overení verejného archívu.
