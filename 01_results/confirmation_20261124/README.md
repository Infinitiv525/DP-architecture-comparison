# Nová nezávislá sada: TFNO, dense FNO a RealNPU

Používateľ 24.11.2026 schválil 20 nových tréningov každej z troch metód, spolu 60. Pôvodných 5 pilotných tréningov sa do primárnej 20-seedovej analýzy nepripočíta. Ide o samostatný cielený dodatok na tom istom pevnom benchmarku, ktorého TEST už bol známy pri plánovaní.

- [Vopred uzavretý protokol](../../03_methodology/confirmation_20261124/protokol.txt)
- [Presný plán, seedy a rozpočet](../../04_source/confirmation_20261124/plan.json)
- [Všetky úlohy a pôvodné konfigurácie](../../04_source/confirmation_20261124/selection.json)
- [Úspešná CPU kontrola všetkých60 inicializácií](cpu_gate.json)
- [Aktuálny zaznamenaný stav](status.json)
- [Odoslanie úloh](submission.json)

Žiadne nové HPO ani warm-start. Rovnakých 640 TRAIN,160 VAL a 200 TEST, rovnaké pôvodné modely a najviac 4 H200-GPU-h na tréning. Každý nový seed sa používa vo všetkých troch metódach; medzi blokmi sú veľké rozostupy, aby sa pri pôvodnom pravidle shuffle seed+epoch neprekrývali epochové náhodné prúdy.

Primárna rodina má 9 porovnaní:3 dvojice × MSE/L2 rýchlosti/L2 tlaku. Primárnou jednotkou je 20 nových tréningových seedov; ide o opakovateľnosť na pevnom benchmarku. Zovšeobecnenie na nové geometrie sa nesľubuje.

Hlavný tréningový strop 240 GPU-h; odhad učenia a TEST 243,42 GPU-h, plánovacia rezerva 260 GPU-h. Celé polia a váhy zostávajú na Perune do priamej archivácie naGitHub. Predbežná príprava v1 mala iba technické kontroly, žiadne hlavné tréningy; aktívny je priečinok v2.

Samostatný auxiliary operational_integrity.json bol prevzatý zo staršej prípravy. Aktuálne celé SHA zdrojov určuje source_integrity.json; CPU brána odkazuje na jeho presný SHA.

## Organizácia odoslania pred hlavným štartom

Plánovač má limit50 odoslaných a20 bežiacich úloh. Všetkých60 tréningov vykonajú automatické dávky40+20. Prvých40 bude v plánovači, CPU úloha po ich ukončení odošle zostávajúcich20 a následný audit. Maximálny súbeh je20 GPU. Matematika, konfigurácie, seedy a4h rozpočet na model sa nemenia.

## Hlavné tréningy spustené

Snapshot 2026-11-24T18:34:51.762214+01:00:prvé pole173839, automatické odoslanie druhej dávky173840. Stav pri overení:{'PENDING': 25, 'RUNNING': 15}. Celkovo60 modelov je zahrnutých v automatickom postupe. [Záznam postupu](dokumentacia.txt). Po tréningoch nasleduje samostatný TEST a štatistika.
