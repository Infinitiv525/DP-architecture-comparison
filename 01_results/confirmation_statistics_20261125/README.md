# Potvrdenie troch architektúr na20nových tréningoch
[Čitateľné vyhodnotenie](VYSLEDKY.txt) · [Všetky primárne testy a intervaly](paired_t_primary.json) · [Nezávislé overenie](independent_verification.json).
TFNO má na pevnom200-prípadovom benchmarku nižšie chyby než denseFNO ajRealNPU vo všetkých3metrikách. Osem z9vopred určených kontrastov prešloHolm korekciou; celkováMSE denseFNO–RealNPU sa nepreukázala.
Používame20nových semien/metódu;pôvodných5pilotných behov je oddelených.
Geometrická bootstrapcitlivosť je slabšia/závisí od variantu intervalu. Všeobecnú prevahu na nevidených geometriách netvrdíme.
Všetkých60modelov splnilo per-casep99<=5ms naH200; jednotlivé prekročenia sú priznané v reporte.
[Kompletný archív a overenie](ARCHIV.txt) · [Váhy a všetky polia podľa architektúry a behu](../../02_models/confirmation20_20261125/README.md).
[Protokol](../../03_methodology/confirmation_statistics_20261125/protokol.txt) · [Zdroj](../../04_source/confirmation_statistics_20261125).
