# Váhy a index sietí

Finálne vybraných modelov: 59. Celkom zachovaných checkpointových súborov: 587.

[Archívy na stiahnutie](https://github.com/Infinitiv525/DP-architecture-comparison/releases/tag/research-artifacts-v1), [index všetkých checkpointov](model_index.csv), [úplný manifest](ARTIFACT_MANIFEST.json), [kontrolné súčty](SHA256SUMS.txt). Každý diel tar.gz sa samostatne rozbaľuje do rovnakého priečinka. Finálne siete sú pod `models/final`, ostatné pod `models/history`. Najprv over SHA dielu a následne SHA požadovaného súboru z manifestu.

`best` je vybraný checkpoint; `last` a `training_state` sú dostupné posledné/obnovovacie stavy. Nie sú to ďalšie nezávislé siete. Označenie final neznamená nulovú chybu. Bezpečne rozbaľuj napríklad cez Python 3.12+ `tarfile` s `filter="data"`. PyTorch checkpointy načítavaj len z dôveryhodného overeného pôvodu.
