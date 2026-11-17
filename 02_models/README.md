# Váhy a index sietí

Finálne vybraných modelov: 59. Celkom zachovaných checkpointových súborov: 587.

[Archívy na stiahnutie](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/research-artifacts-v1), [index všetkých checkpointov](model_index.csv), [úplný manifest](ARTIFACT_MANIFEST.json), [kontrolné súčty](SHA256SUMS.txt). Každý diel tar.gz sa samostatne rozbaľuje do rovnakého priečinka. Finálne siete sú pod `models/final`, ostatné pod `models/history`. Najprv over SHA dielu a následne SHA požadovaného súboru z manifestu.

`best` je vybraný checkpoint; `last` a `training_state` sú dostupné posledné/obnovovacie stavy. Nie sú to ďalšie nezávislé siete. Označenie final neznamená nulovú chybu. Bezpečne rozbaľuj napríklad cez Python 3.12+ `tarfile` s `filter="data"`. PyTorch checkpointy načítavaj len z dôveryhodného overeného pôvodu.

## Vyber konkrétnu rodinu alebo metódu

| Metóda | Vybrané záznamy | Váhy a seedy |
|---|---:|---|
| CasCor | 5 | [Otvoriť modely](cascor/README.md) |
| ChebKAN | 5 | [Otvoriť modely](chebkan/README.md) |
| ChebKAN — druhá šanca | 2 | [Otvoriť modely](chebkan_second_chance/README.md) |
| Dendrite | 5 | [Otvoriť modely](dendrite/README.md) |
| ELM | 5 | [Otvoriť modely](elm/README.md) |
| Dense FNO | 5 | [Otvoriť modely](fno/README.md) |
| PiNet | 5 | [Otvoriť modely](pinet/README.md) |
| Pomocná PINN | 5 | [Otvoriť modely](pinn/README.md) |
| RBF | 5 | [Otvoriť modely](rbf/README.md) |
| RBF — druhá šanca | 2 | [Otvoriť modely](rbf_second_chance/README.md) |
| RealNPU | 5 | [Otvoriť modely](realnpu/README.md) |
| SCN | 5 | [Otvoriť modely](scn/README.md) |
| TFNO | 5 | [Otvoriť modely](tfno/README.md) |
