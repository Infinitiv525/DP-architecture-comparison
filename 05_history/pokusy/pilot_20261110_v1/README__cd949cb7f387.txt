TECHNICKÝ PILOT VYBRANÝCH ARCHITEKTÚR NA PERUNE
10.11.2026. Oddelený od dokončenej LLM/PINN štúdie.

ČO TERAZ OVERUJEME
Či vieme všetkých10 položiek výberu v6 spustiť na rovnakej GPU, získať
platné celé pole512x512, trénovať nové modely bez NaN a merať presnosť,
fyzikálne diagnostiky, čas a pamäť. Krátky pilot nie je záverečné porovnanie
architektúr a nedovoľuje tvrdiť štatistickú nadradenosť ani hotovú presnosť.

MODELY
Autorské FNO/TFNO; Chebyshev KAN; Π-Net; Dendrite Net; RealNPU; RBF;
Cascade-Correlation; ELM; SCN; naša archívna PINN fixed_k00.
Wavelet NN a F-INR sú vyradené. Autorský TFNO wrapper pochádza z
https://github.com/baskargroup/flowbench-tools/blob/6877099b8883232138c73b2b6056ff01f7d3697b/Training/codes/models/FNO.py
Používame neuraloperator==2.0.0, explicitný Tucker rank0.1,5 vrstiev,
32 módov a32 hidden kanálov podľa príkladu autorského README. API2.0 používa
projection_channel_ratio=4 na požadovaných128 projekčných kanálov.
Vstupy/výstupy a normalizácia sú adaptované na náš split; nejde o identickú
replikáciu čísel článku alebo slepé kopírovanie rozdielnych autorských príkladov.

DÁTA A ROZSAH
Prvých16 TRAIN a4 VAL prípadov z existujúceho zmrazeného rozdelenia FlowBench
LDC harmonics; žiadny výber podľa ich cieľových hodnôt. Plné natívne polia
u,v,p512x512, rovnaká fyzikálna škála a tlaková referencia. TEST target súbory
sa neotvárajú. Pôvodná štúdia už TEST otvorila; tento pilot nie je nový slepý test.
Normalizácia je z pôvodných640 TRAIN, bez VAL/TEST. Nemení sa rozdelenie dát.

ÔSME ALTERNATÍVY A SPOLOČNÁ GEOMETRIA
Pre prvý technický pilot majú bodové alternatívy rovnaké deterministické
príznaky: súradnice/Fourier, logRe, lokálna maska/SDF a4x4 pooling geometrie.
Nie je použitý už naučený encoder zo starej PINN ani neoznačený tréning navyše.
Toto je jednoduchá dočasná reprezentácia, nie zmrazená finálna voľba encoderu.
TFNO vidí celé známe vstupné pole; preto výsledok nie je izolovaný účinok
výmeny jednej vrstvy. Rozdiely reprezentácie a tréningu sú výslovne priznané.

FNO a5 gradientových alternatív dostanú120 Adam aktualizácií s dátovou stratou.
Bodové modely používajú2048 spoločných vzoriek na TRAIN prípad; FNO celé polia.
ELM rieši globálny ridge fit, SCN prijíma kandidátov podľa reziduálnej podmienky,
CasCor učí kandidátov podľa korelácie s chybou. SCN/ELM limit32 neurónov,
CasCor32 pridaných neurónov. Sú to malé skúšobné konfigurácie; nie sú
parametrovo ani tréningovou prácou presne zrovnané. Po pilotovi ich treba
upraviť pred riadnym opakovaným výskumným porovnaním.

Pôvodná PINN sa iba načíta z vopred určenej fixed_k00 referencie s overeným
SHA256. Má400 pôvodných epoch a640TRAIN; jej presnosť sa nesmie interpretovať
ako férové víťazstvo proti krátkym novým tréningom. Žiadny výber podľa TEST.
Nové gradientové alternatívy sú dátové siete, nepremenúvajú sa na PINN bez
fyzikálnej straty. Fyzika sa pri nich zatiaľ meria ako diagnostika.

MATEMATICKÉ A PRAKTICKÉ KONTROLY
checks.py overuje známe polynómy Π-Net/DD, Chebyshev T3, znamienkové delenie
RealNPU, konečné gradienty, skutočné fitovanie/obnovu ELM/SCN/CasCor,
tlakovú gauge invarianciu a TFNO forward/backward. PiNet používa CP-style
reziduálne násobenie; DD fixne vkladá embedded vstup v každom stupni.
RealNPU zachováva relevance gate podľa eq14/15, s explicitnou afinnou
výstupnou adaptáciou. SCN je SC-III-style s priznanou ridge regularizáciou.
Nejde o tvrdenie, že ide o presné originálne implementácie všetkých autorov.

MERANIE A VÝSTUPY
Každý model predikuje všetky4 VAL polia, s nulami v známom telese a vlastným
fluidným priemerom tlaku. GPU latency:20 warmup +100 meraní na prípad;
navyše30 synchronizovaných meraní wall času. Report p50/p95/p99/max a všetky
surové časy. P99<=5ms sa posudzuje pre celé pole, nie jeden bod či interný blok.
V čase je zahrnutá vstupne závislá reprezentácia, sieť, maska a tlaková gauge;
vstupy sú už na GPU. Prenos do PC, vykresľovanie a statické súradnice sú mimo.
100 opakovaní je orientačný pilot, nie konečné spoľahlivé meranie chvosta p99.
Diagnostiky: fyzikálne all-cell u/v/p MSE, fluidné relatívne L2, divergencia
a NS rezíduum v bezpečnom vnútornom stencile, aj rezíduum CFD referencie.
Žiadne výstupné klipovanie alebo vymýšľanie metrík pri neúspechu.

Na Perune ostanú váhy, plné predikcie, raw timings, tréningová história,
parametre, prostredie a zdroje. comparison.png obsahuje CFD/model/chybu;
overview.csv a summary.json sumarizujú aj neúspechy. Do PC sa môžu kopírovať
iba malé tabuľky/JSON/TXT/obrázky, nie váhy alebo veľké numerické polia.

SPUSTENIE A IZOLÁCIA
Nové prostredie envs/architecture-pilot-20261110-v1 dedí hotový Torch,
nové závislosti inštaluje do vlastného venv. Pôvodné prostredie sa nemení.
deploy.py vytvorí hashovanú nemennú revíziu zdrojov, overí prenos a matematické
kontroly. --submit zadá jedinú GPU úlohu; rovnaký source hash sa znovu nezadá.
Všetkých10 modelov beží po jednom v čerstvom procese na1 H200. Limit jedného
modelu600s, scheduler limit02:00:00. Timeout je neúspech, nie falošná chyba0.
Slurm accounting po skončení poskytne skutočne pridelené GPU/core sekundy.
Výsledky: /mnt/project/perun2601449/dp_llm_cfd/results/architecture_pilot_20261110_v1/job_JOBID/
Táto úloha automaticky nespúšťa rozsiahly experiment ani LLM ladenie.
