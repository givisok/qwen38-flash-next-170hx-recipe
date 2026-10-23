# Qwen3.8-Flash-Next (AWQ-g32, Uncensored) на 3× CMP 170HX — полный рецепт

Домашний LLM-стенд: **Qwen3.8-Flash-Next 180B-A6B (Uncensored, AWQ-g32)** на vLLM
с пайплайн-параллелизмом на 3 картах **CMP 170HX (unlocked 64 ГБ, sm80/Ampere)**,
1M-токен контекст, MTP-спекулятивный декодинг, KV-кэш fp8.

Рабочая точка: ~120 tok/s одиночный поток, ~45 tok/s на запрос при 16 параллельных
(~600 tok/s агрегат), prefill до 7 900 tok/s на 79K-токенном промпте.

---

## 1. Источники — что и откуда бралось

| Компонент | Откуда | Что взято |
|---|---|---|
| Базовый стек (форк vLLM + docker-образ + лаунчер) | [clee/qwen3.8-flash-next-3x170hx](https://github.com/clee/qwen3.8-flash-next-3x170hx) | Форк `vllm-sm80` (линейия `q38fn/stack` поверх пиннингованного nightly `7d0b4e57`), скрипт запуска, устройство PLE-стриминга с NVMe (preadv), YaRN-override, тиринг (позже отключён — см. §6) |
| Модель (чекпоинт) | [leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) (~137 ГБ, 87 shards) | AWQ W4A16 **group-32** (шаг 32 выбран под ширину эксперта 640: 640/4 ранга = 160, делится на 32, но не на 128), damped GPTQ на 169K строк реальных активаций, KL 0.0214 / top-1 97.08% против BF16; PLE-таблица в **fp8** (51 ГБ вместо 95 ГБ BF16); калиброванные QSA KV-шкалы; MTP-черновик в комплекте; uncensored-база [orcarouter](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored) |
| Тюнинг под это же семейство | [mannydelrio/qwen38-flash-next-2x-cmp170hx](https://github.com/mannydelrio/qwen38-flash-next-2x-cmp170hx) | Идея явных **cudagraph capture sizes с шагом 4** до 64 (vLLM округляет MTP-батч до (1+3)=4 токена; без этого хвост 16-поточного декода работает без графов). Их INT4-PLE (tinybeast-плагины) непереносим и не нужен: PLE и так стримится с NVMe |
| Предыдущая модель (для сравнения/отката) | [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | Работала без патчей; крупнее (169 ГБ), PLE в BF16, медленнее prefill |
| Chat-шаблон | чекпоинт leoncca (`chat_template.jinja`) | thinking-режим, reasoning effort **xhigh/medium/low** (другие значения шаблон отвергает с 400) |

Альтернативы проверены и отвергнуты: NVFP4-квант (lambsea) — требует Blackwell;
GGUF/MLX/EXL3 — не vLLM; Mochgolf W4A16-INT8PLE — тот же размер, качество не
измерено, проверен только в чужом SGLang-форке.

## 2. Железо и контейнер

- Proxmox VE, хост с 3× NVIDIA CMP 170HX (unlock до 64 ГБ = sm80), 32 ГБ RAM, NVMe 2 ТБ.
- LXC-контейнер **CT 101** (Debian, unprivileged, `nesting=1 keyctl=1`):
  - memory 26000, **swap: 0** (инференсу подкачка только вредит: tmpfs/KV-тир
    начинает уходить на диск; см. §6 «своп»),
  - проброс `/dev/nvidia0-2`, `nvidiactl`, `nvidia-uvm*`, `lxc.prlimit.memlock: unlimited`,
  - `onboot: 1`, диск 1.4 ТБ.
- Внутри: docker, образ `vllm-sm80:q38fn` (собран из Dockerfile репо clee поверх
  пиннингованного nightly), дерево форка в `/root/src/vllm-sm80`, модели в
  `/home/models` (смонтированы в контейнер read-only как `/models`).

## 3. Ключевые решения конфигурации

Финальный serve-конфиг: [`configs/awq-1m-fp8-ssmbf16.yml`](configs/awq-1m-fp8-ssmbf16.yml).

| Параметр | Значение | Почему |
|---|---|---|
| `pipeline-parallel-size` | 3 | P2P между картами отключён (глюки DMA на этом железе) → TP невозможен, только PP |
| `VLLM_PP_LAYER_PARTITION` | **17,16,15** | 48 слоёв; замер утилизации/мощности по рангам показал: у последнего ранга + MTP-черновик — он самый тяжёлый, ему меньше слоёв. Свип 16/16/16 → 17/16/15 дал +7.5% одиночного потока; 18/16/14 хуже |
| `speculative-config` | `method: mtp, num_speculative_tokens: 3` | Черновик едет в чекпоинте; +75% одиночному потоку (120 vs 66-70 без MTP). ngram_gpu пробовали — хуже |
| `max-num-seqs` | 16 | 8→16 подняло агрегат с 360 до ~500-600 tok/s, карты загружены на 84-100% |
| `cudagraph_capture_sizes` | `[1,2,4,...,64]` шаг 4 | из репо mannydelrio: батч MTP округляется к 4 |
| `kv-cache-dtype` | fp8_e4m3 | софтверная эмуляция на sm80; KV-пул 4.57M токенов |
| `max-model-len` | 1 048 576 | YaRN ×4 от 262144 (в `hf-overrides`, + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`), `block-size: 1728`, `rope_theta 1e7` |
| `layer_types` override | 3× linear_attention + 1× qwen_sparse_attention ×12 | мост трансформерс-ренейминга (full_attention → indexed_attention) |
| `NCCL_P2P_DISABLE=1` + `disable-custom-all-reduce` | | P2P виснет на этих картах |
| `gpu-memory-utilization` | 0.94 | потолок без OOM; KV-пул 4 573 327 токенов, конкарренси 4.36× на 1M |
| **kv-transfer-config (тиринг)** | **ОТСУТСТВУЕТ (вырезан)** | см. §6 — источник порчи KV |

## 4. Патчи форка под этот чекпоинт (обязательны)

Лежат в [`patches/`](patches/) и вшиты в дерево `/root/src/vllm-sm80`
(лаунчер бинд-маунтит файлы форка поверх образа):

1. **`mtp-weights-remap.diff`** — чекпоинт везёт QSA KV-шкалы
   (`mtp.layers.0.self_attn.{q,k,v}_scale`). AutoWeightsLoader прогоняет их через
   `quant_config.get_cache_scale_mapper()`, который переписывает `X.k_scale` в
   `X.attn.k_scale` (раскладка stock-внимания) — у QSAAttention ребёнка `attn` нет →
   краш «no module or parameter named 'attn'». Фикс: k/v → на буферы владельца
   (`_k_scale`/`_v_scale`), q_scale отбрасывается.
2. **`ple-fp8-staging.diff`** — PLE-стейджинг был захардкожен под BF16-строки
   (`head_dim*2` байт). У fp8-таблицы строки `head_dim*1` = 160 Б → краш
   320-vs-160 на первом warmup. Фикс: размер буфера от dtype таблицы, на forward
   — вью + поэлементный деквант fp8→bf16 (кадрируемый кудаграфом kernel).
3. **`VLLM_PLE_SYNC=1`** (env в лаунчере) — форковый код сам подозревал гонку
   («to isolate whether the corruption is a staging race»): async-копия H2D
   PLE-строк из pinned-памяти против старта forward. Перевод таблицы на fp8
   сместил тайминги — гонка открылась. Sync закрывает.

## 5. Запуск и операции

```bash
# на CT 101, из /root/qwen38fn
cd /root/qwen38fn && IMAGE=vllm-sm80:q38fn FORK=/root/src/vllm-sm80 ./run-qwen38fn.sh   # запуск
./switch-to-awq.sh    # смена модели: rm контейнера, чистка /dev/shm + kv-disk, запуск, restart-policy
./clean-kv.sh         # лекарство от порчи KV (до отключения тиринга): вытереть kv-disk + рестарт
```

- Docker RestartPolicy — **unless-stopped** (иначе после выключения питания хоста
  сервер не поднимался — главная причина «умерло после перезагрузки»).
- Лаунчер чистит `/dev/shm/vllm_offload_*.mmap` и psm_* от прошлого килла —
  неснесённый 8 ГБ mmap блокирует аллокацию CPU-тира.
- Эндпоинт: `http://<ip-ct>:8080/v1`, модель зовётся `qwen3.8`. Клиенты: pi-агент
  (`~/.pi/agent`) и Hermes Agent (CT 102, `reasoning_effort: medium` в его
  `config.yaml`; допустимы только xhigh/medium/low — остальное шаблон отвергает).

## 6. Грабли, на которые наступили (и их решения)

1. **Своп в LXC**: у CT подкачка забивалась на 100% (4 ГБ) — KV/tmpfs уходил на
   диск, латентность умирала. Лечение: `pct set 101 --swap 0` + `vm.swappiness=10`
   на хосте + `swapoff -a && swapon -a` для дрена. Инференсу своп не нужен.
2. **«Белиберда» — мультиязычный токен-суп** (`sm exit asset 资产...`):
   - НЕ модель, НЕ шаблон, НЕ reasoning_effort. Разные жертвы по мере расследования:
     сначала выглядело как effort=medium (маскировка: разные effort → разные хэши
     префикса → разные блоки в тире), затем как PLE-гонка (реальная, закрыта sync).
   - **Финальный корень: KV-оффлоад-тиринг** (OffloadingConnector: CPU 8 ГБ + NVMe
     fs-тир) — форковый код возвращал битые блоки для многочанковых кэшированных
     префиксов. Доказано абляцией: один и тот же промпт — 5/5 мусор с тирингом,
     6/6 чисто без, затем 0/32 мусора в 8-поточной батарее без тиринга.
   - Тиринг **вырезан из конфига** (бэкап `.bak-tiering` внутри CT). GPU-префикс-кэш
     остался, скорость декода не изменилась (45.0 vs 44.9 tok/s на запрос).
     Потеряно: тёплый кэш префиксов, переживший рестарт. Вернуть — только чиня
     кандидаты из `patches/NOTES.md` (гонка lookup/eviction в fs/manager.py,
     смещения чанков в submit_store, исключение draft-групп в spec.py).
3. **MTP-черновик не грузился** на новом чекпоинте — патч №1 (§4).
4. **PLE в fp8** не влезал в стейджинг — патч №2 (§4).
5. **Чистить kv-disk при смене модели**: fs-тир адресуется по токенам, значения —
   от старых весов (сейчас тиринг выключен, но правило помним на будущее).
6. **OOM-стресс не роняет сервер**: 8×249K и 5×947K токенов (переподписка пула) —
   всё в очереди, oom_kill=0; скрипт `scripts/oomtest.py`. Но именно такой churn
   загонял тир в eviction и провоцировал п.2.

## 7. Замеры (AWQ-g32, финал, против Intel AutoRound W4A16)

| Тест | Intel W4A16 | AWQ-g32 |
|---|---|---|
| Одиночный поток (TG) | 115–121 tok/s | 117–129 tok/s |
| 16 параллельных, на запрос | 45.4 | 44.9–45.0 |
| 16 параллельных, агрегат | 561 | 514–604 (шум длины генераций) |
| Prefill холодный @79K ток. | 7 436 tok/s | 7 880–7 920 tok/s |
| Echo (копирование текста) | 138.2 tok/s | 148.4 tok/s |
| KV-пул GPU | 4.45M | 4.57M токенов |
| Чекпоинт на диске | 169 ГБ | 133 ГБ (PLE 51 ГБ fp8) |

Скрипты замеров: `scripts/verify16.sh` (параллельная нагрузка + GPU-утилизация),
`scripts/prefill.py` (PP/TG), `scripts/oomtest.py` (стресс-память).

## 8. Порядок воспроизведения с нуля (кратко)

1. Proxmox: создать CT (§2), пробросить карты, `swap 0`, поставить docker.
2. Забрать репо clee (форк + Dockerfile), собрать `vllm-sm80:q38fn`,
   checkout ветки `q38fn/stack` в `/root/src/vllm-sm80`.
3. Скачать чекпоинт leoncca AWQ-g32 (137 ГБ, проверить SHA256SUMS) в
   `/home/models/qwen38fn-awq-g32`; chat_template → в конфиги лаунчера.
4. Наложить патчи §4, положить `configs/` и `scripts/` из этого репо.
5. `docker update --restart unless-stopped` после первого запуска.
6. Проверить: одиночный запрос → N=16 (`verify16.sh`) → prefill (`prefill.py`) →
   на длинных агентских промптах с тулзами и историей (мульти-таходность!).
