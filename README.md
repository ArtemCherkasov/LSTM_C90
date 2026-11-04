# LSTM

---

## 🇷🇺

Проект реализует LSTM (Long Short-Term Memory) нейронную сеть на стандарте C90 для прогнозирования котировок EUR/USD. Работает в двух режимах: **OHLC** (прогноз цен Open/High/Low/Close) и **Direction** (прогноз направления изменения цены).

### Требования

- Компилятор C с поддержкой C90 (gcc, clang, MSVC и др.)
- CMake 3.10+
- математическая библиотека (libm, подключается автоматически)

### Сборка

```bash
# Основной бинарник + тесты
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Только тесты
make test_lstm

# Только основной бинарник
make LSTM_C90
```

### Запуск тестов

```bash
./test_lstm
# Ожидаемый результат: Results: 31/31 tests passed
```

### Формат данных

CSV-файл в формате экспорта MetaTrader (таб-разделитель, поля в двойных кавычках):

```
<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
2009.06.12\t00:00:00\t1.40954\t1.41129\t1.40929\t1.41069\t1002\t0\t20
```

- **Таймфрейм:** H1 (часовые бары)
- **Период данных:** 2009-06-12 … 2025-09-25
- **Используемые столбцы:** OPEN, HIGH, LOW, CLOSE, TICKVOL
- **Игнорируемые:** VOL (всегда 0), SPREAD (не используется)
- Для замены данных — экспортируйте CSV из MetaTrader в том же формате

### Архитектура

```
Вход (26 признаков × 64 шага) → LSTM (128 ячеек) → LINEAR (128→num_outputs) → predict_count свечей
           many-to-many: output на каждом шаге
```

- **Стандарт языка:** C90 (без C99-фич)
- **LSTM ячеек:** 128 (размер hidden state, не зависит от длины последовательности)
- **Признаков:** 26 (13 базовых + 13 мульти-таймфрейм, на каждый временной шаг)
- **Временных шагов:** 64 (длина входной последовательности)
- **Выходов:** `predict_count × 4` (OHLC) или `predict_count` (Direction), на каждый шаг
- **Архитектура:** many-to-many — output layer срабатывает на каждом шаге
- **Горизонт прогноза по умолчанию:** 1 свеча (`--output 1`), можно увеличить через CLI
- **Функция потерь:** MSE (OHLC) или BCE (Direction)
- **Авторегрессия:** нет (один forward pass → сразу все свечи)

### Режимы работы

**OHLC** (`--mode ohlc`, по умолчанию): прогноз будущих цен OHLC как returns.
На каждом шаге `t` LSTM предсказывает `predict_count` будущих свечей OHLC. Метки для шага `t` — returns следующих свечей относительно `close_t`. Выходы: `predict_count × 4` (4 OHLC values на свечу).

**Direction** (`--mode direction`): прогноз направления изменения цены (вверх/вниз).
Метки: `1.0` если `close_{t+k} > close_t`, иначе `0.0` (бинарная классификация). Выходы: `predict_count` (1 направление на шаг). Выходная активация: **sigmoid** (выход в (0,1)). Функция потерь: **BCE** (Binary Cross-Entropy). Threshold: `pred > 0.5` → вверх, иначе → вниз.

### Обучение (many-to-many)

Loss суммируется по всем `seq_len × num_outputs` выходам, что даёт больше градиентного сигнала на каждый семпл. Обучение: mini-batch SGD + momentum, градиентный clipping по элементам в [-5, +5].

**Ctrl+C** во время обучения — модель сохраняется корректно (signal handler).

### Предсказание

Один forward pass → берём выходы **последнего шага** (`outputs[(seq_len-1) × num_outputs + o]`) → денормализация → предсказанные значения.
- **Без авторегрессии:** один forward pass → сразу все свечи

### Признаки (26)

**13 базовых признаков** (индексы 0–12):

| # | Имя | Описание |
|---|-----|----------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1}, обрезка [-10,10] |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 баров) |
| 6 | `rsi` | RSI(14) / 100, [0,1] |
| 7 | `macd_hist` | (EMA12 - EMA26) - EMA9(сигнал) |
| 8 | `bb_position` | (close - BB_lower) / (BB_upper - BB_lower), BB = MA20 ± 2σ |
| 9 | `atr_norm` | ATR(14) / close |
| 10 | `stoch_k` | Stochastic %K(14) / 100, [0,1] |
| 11 | `hour` | hour_of_day / 23 |
| 12 | `dow` | day_of_week / 6 |

**13 мульти-таймфрейм признаков** (индексы 13–25): те же 13 признаков, вычисленные на H2-агрегированных данных (каждые 2 H1 свечи → 1 H2 свеча), наложенные на H1 строки.

| # | Имя | Описание |
|---|-----|----------|
| 13 | `mtf_candle_return` | candle_return на H2 |
| 14 | `mtf_candle_range` | candle_range на H2 |
| 15 | `mtf_close_return` | close_return на H2 |
| 16 | `mtf_volume_return` | volume_return на H2 |
| 17 | `mtf_ma_diff` | ma_diff на H2 |
| 18 | `mtf_volatility` | volatility на H2 |
| 19 | `mtf_rsi` | rsi на H2 |
| 20 | `mtf_macd_hist` | macd_hist на H2 |
| 21 | `mtf_bb_position` | bb_position на H2 |
| 22 | `mtf_atr_norm` | atr_norm на H2 |
| 23 | `mtf_stoch_k` | stoch_k на H2 |
| 24 | `mtf_hour` | hour на H2 |
| 25 | `mtf_dow` | dow на H2 |

### Выходы (OHLC режим)

Для каждой из `predict_count` будущих свечей (k=1..predict_count) предсказываются 4 returns относительно `ref_close` (close последнего бара последовательности):

| Индекс | Выход | Формула |
|--------|-------|---------|
| k·4+0 | open_return_k | (open_{t+k} - ref_close) / ref_close |
| k·4+1 | high_return_k | (high_{t+k} - ref_close) / ref_close |
| k·4+2 | low_return_k | (low_{t+k} - ref_close) / ref_close |
| k·4+3 | close_return_k | (close_{t+k} - ref_close) / ref_close |

Денормализация: `price = ref_close × (1 + denorm(pred))`

### Выходы (Direction режим)

Для каждого из `predict_count` будущих шагов (k=1..predict_count): 1 значение — вероятность направления вверх.

| Индекс | Выход | Формула |
|--------|-------|---------|
| k | direction_k | 1.0 если close_{t+k} > close_t, иначе 0.0 |

Threshold: `pred > 0.5` → вверх (1), иначе → вниз (0).

### Агрегация таймфрейма (`--h`)

Параметр `--h N` группирует каждые N последовательных H1 свечей в одну, создавая более высокий таймфрейм. Все вычисления (признаки, метки, обучение, предсказание) работают с агрегированными данными.

Правила агрегации N свечей H1 в одну:

| Поле | Значение |
|------|----------|
| Open | open первой свечи в группе |
| High | max(high) по всем свечам в группе |
| Low | min(low) по всем свечам в группе |
| Close | close последней свечи в группе |
| TickVol | sum(tickvol) по всем свечам |
| Date/Time | первой свечи |

Неполная последняя группа отбрасывается.

| Параметр | Строк данных | Описание |
|----------|-------------|----------|
| `--h 1` (default) | ~101,000 | исходные H1 свечи |
| `--h 2` | ~50,500 | H2 — агрегация по 2 свечи |
| `--h 4` | ~25,250 | H4 — агрегация по 4 свечи |
| `--h 8` | ~12,600 | H8 — агрегация по 8 свечей |

### Параметры

**CLI-регулируемые:**

| Флаг | Дефолт | Описание |
|------|--------|----------|
| `--cells` | 128 | Кол-во LSTM ячеек (размер hidden state) |
| `--epochs` | 100 | Кол-во эпох обучения |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Длина входной последовательности (баров) |
| `--mode` | `ohlc` | Режим: `ohlc`, `direction`, `ohlcpredict`, `directionpredict`, `realdata` |
| `--output` | 1 | Кол-во выходов (OHLC векторов или шагов направления) |
| `--datetime` | — | `YYYY.MM.DDThh:mm:ss` — дата-время для прогноза (обязательно для predict-режимов) |
| `--from` | — | `YYYY.MM.DDThh:mm:ss` — начало диапазона для `--mode realdata` |
| `--to` | — | `YYYY.MM.DDThh:mm:ss` — конец диапазона для `--mode realdata` |
| `--data` | `../resources/EURUSD_*.csv` | Путь к CSV-файлу |
| `--model` | `../resources/model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt` | Путь к файлу модели |
| `--h` | 1 | Агрегация таймфрейма: группировка N свечей H1 в одну (H1=1, H2=2, H4=4, ...) |

**Зашиты в код** (в `lstm.h` / `main.c`):

| Параметр | Значение | Где |
|----------|----------|-----|
| `OUTPUTS_PER_CANDLE` | 4 (OHLC) | `lstm.h` |
| `DEFAULT_OHLC_PREDICT_COUNT` | 1 | `lstm.h` |
| `PREDICTION_THRESHOLD` | 0.5 | `lstm.h` |
| Batch size | 32 | `main.c` |
| Momentum (SGD) | 0.9 | `main.c` |
| Seed | 42 | `main.c` |
| Weight decay (L2) | 1e-6 | `lstm.c` |
| Gradient clipping | [-5.0, +5.0] | `utils.h` |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) | `lstm.c` |
| Forget gate bias init | 1.0 (Jozefowicz et al., 2015) | `lstm.c` |
| Train/test split | 80/20 chronological | `data.c` |
| `NUM_RAW_FEATURES` | 26 | `data.h` |
| `NUM_BASE_FEATURES` | 13 | `data.h` |
| `NUM_MTF_FEATURES` | 13 | `data.h` |

### Режимы работы

**1. Обучение** (`--mode ohlc` или `--mode direction`):
```bash
./LSTM_C90 --mode ohlc --cells 128 --epochs 100 --lr 0.005 --seq 64 --output 1
./LSTM_C90 --mode ohlc --cells 64 --epochs 50 --lr 0.001 --seq 32 --output 5
./LSTM_C90 --mode direction --epochs 50 --lr 0.01 --output 3
./LSTM_C90 --mode direction --epochs 50 --lr 0.01 --output 3 --h 4
```
Модель сохраняется в `model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt`. **Ctrl+C** — корректное сохранение модели.

**2. Прогноз** (`--mode ohlcpredict` или `--mode directionpredict`):
```bash
./LSTM_C90 --mode ohlcpredict --datetime 2024.01.15T08:00:00 --output 5
./LSTM_C90 --mode directionpredict --datetime 2024.01.15T08:00:00 --output 3
./LSTM_C90 --mode ohlcpredict --datetime 2024.01.15T08:00:00 --output 5 --h 4
./LSTM_C90 --mode directionpredict --datetime 2024.01.15T08:00:00 --output 3 --h 4
```
Если `predict_count`, `mode`, `seq` или `--h` не совпадают с моделью — ошибка и выход. Один forward pass → `num_outputs` выходов → денормализация → предсказанные значения. Выводятся только строки прогноза в количестве `--output`. **Без авторегрессии.**

**3. Реальные данные** (`--mode realdata`):
```bash
# Все котировки из файла
./LSTM_C90 --mode realdata
# С указанной даты до конца файла
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00
# В промежутке от --from до --to
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00 --to 2024.02.15T08:00:00
# С агрегацией таймфрейма H4
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00 --to 2024.02.15T08:00:00 --h 4
```
Вывод: `YYYY.MM.DDThh:mm:ss O H L C TICKVOL` — по одной строке на свечу. Агрегация `--h` применяется так же, как при обучении и прогнозе.

### Формат файла модели

Текстовый формат, читаемый человеком:

```
# LSTM Model - Text Format
# num_cells: 128
# num_features: 26
# num_outputs: 4
# predict_count: 1
# mode: ohlc
# output_activation: 0
# seq_len: 64
# epochs_trained: 100
# best_train_mse: ...
# best_test_direction_accuracy: ...

[normalization]
feature_name: candle_return | min: ... | max: ...
...
label_min: ... ...
label_max: ... ...

[lstm_weights]
# Gate: forget
forget_recurrent:
  row[0]: w0 w1 ... w_combined
...
forget_input:
  row[0]: ...
forget_bias: b0 b1 ... b_n

[output_weights]
weight_output_layer:
  row[0]: w0 w1 ... w_n
bias_output_layer: b0 b1 ... b_num_out
```

Веса сохраняются с полной точностью double (`%.17g`). `output_activation`: 0=linear (OHLC), 1=sigmoid (Direction).

### Структура проекта

```
├── CMakeLists.txt               # CMake конфигурация (C90)
├── main.c                       # Точка входа, CLI, цикл обучения
├── include/
│   ├── types.h                  # LSTMCellState
│   ├── utils.h                  # Матрицы, математика, random
│   ├── data.h                   # DataRow, Dataset (общие)
│   ├── lstm.h                   # LSTMNetwork, константы
│   ├── model.h                  # Сохранение/загрузка модели
│   ├── ohlc/
│   │   ├── data_ohlc.h          # data_build_dataset_ohlc, normalize_labels_ohlc
│   │   └── predict_ohlc.h       # predict_ohlc, demo_ohlc
│   └── direction/
│       ├── data_direction.h     # data_build_dataset_direction, normalize_labels_direction
│       └── predict_direction.h  # predict_direction, demo_direction
├── src/
│   ├── utils.c, data.c, lstm.c, model.c   # Общие
│   ├── ohlc/
│   │   ├── data_ohlc.c          # OHLC returns как labels
│   │   └── predict_ohlc.c       # Денормализация в цены
│   └── direction/
│       ├── data_direction.c     # 0/1 direction labels
│       └── predict_direction.c  # Threshold 0.5 (0/1)
├── tests/
│   ├── test_utils.h/.c, test_main.c       # Общие хелперы, runner
│   ├── test_forward.c, test_backward.c, test_update.c   # Общие LSTM тесты
│   ├── test_model_io.c, test_deterministic.c, test_gradient.c
│   ├── test_features_26.c                 # Тесты с 26 признаками
│   ├── ohlc/
│   │   ├── test_training_ohlc.c     # Training loop (OHLC labels)
│   │   ├── test_data_ohlc.c        # Volume clip & label norm
│   │   └── test_deterministic_ohlc.c # Forward+BPTT vs scalar (linear)
│   └── direction/
│       ├── test_data_direction.c   # Labels 0/1
│       ├── test_training_direction.c # Training loop (sigmoid+BCE)
│       ├── test_sigmoid_gradient.c  # Gradient check (sigmoid+BCE)
│       └── test_deterministic_direction.c # Forward+BPTT vs scalar (sigmoid+BCE)
└── resources/
    ├── EURUSD_H1_*.csv
    └── model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt
```

### API — описание всех функций

#### `include/lstm.h` / `src/lstm.c`

**Константы:**
- `OUTPUTS_PER_CANDLE = 4` — OHLC
- `DEFAULT_OHLC_PREDICT_COUNT = 1` — горизонт прогноза по умолчанию
- `PREDICTION_THRESHOLD = 0.5` — threshold для direction
- `ACTIVATION_LINEAR = 0`, `ACTIVATION_SIGMOID = 1`

**Тип `LSTMNetwork`** — содержит для каждого gate (forget, input, output, candidate):
- `weight_*_recurrent` — матрица [n × (n+f)] (recurrent + input веса)
- `weight_*_input` — матрица [n × f] (зарезервировано)
- `bias_*` — вектор [n]
- `dweight_*`, `dbias_*` — градиенты
- `mweight_*`, `mbias_*` — momentum

Output layer: `weight_output_layer` [num_outputs × n], `bias_output_layer` [num_outputs].

| Функция | Описание |
|---------|----------|
| `LSTMNetwork *lstm_init(int num_cells, int num_features, int num_outputs, unsigned int seed)` | Создание сети с Xavier/Glorot инициализацией. |
| `void lstm_free(LSTMNetwork *net)` | Освобождение всех ресурсов сети. |
| `void lstm_forward(LSTMNetwork *net, const double *sequence, LSTMCellState *cell_states, int seq_len, double *outputs)` | **Прямое распространение.** Заполняет `outputs[0..num_outputs-1]`. Для каждого шага: forget/input/output/candidate gates, `c_t = f_t·c_{t-1} + i_t·g_t`, `h_t = o_t·tanh(c_t)`. Output layer: `outputs[o] = bias[o] + Σ W_out[o][c]·h[c]`. |
| `void lstm_backward(LSTMNetwork *net, const double *sequence, const LSTMCellState *cell_states, const double *predictions, const double *targets, int seq_len)` | **BPTT.** `d_logit[o] = pred[o] - target[o]` для каждого выхода. `dhidden[c] = Σ_o d_logit[o]·W_out[o][c]`. Дальше — стандартный BPTT. |
| `void lstm_zero_gradients(LSTMNetwork *net)` | Обнуление всех градиентов. |
| `void lstm_update(LSTMNetwork *net, double lr, double momentum, int batch_size)` | SGD + momentum. Градиенты нормализуются на batch_size, обрезаются в [-5, +5]. |
| `double lstm_compute_loss(LSTMNetwork *net, Dataset *ds)` | MSE (OHLC) или BCE (Direction): `Σ loss(pred, target) / (N·seq_len·num_outputs)`. |
| `double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds)` | Direction accuracy: доля `sign(pred) == sign(label)` по всем выходам. |
| `void lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len, double *outputs)` | Инференс: один forward pass, заполняет `outputs`. |

#### `include/data.h` / `src/data.c`

| Функция | Описание |
|---------|----------|
| `int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count)` | Загрузка CSV (формат MetaTrader, таб-разделитель). |
| `double **data_compute_raw_features(...)` | Feature engineering: 26 признаков на бар (13 базовых + 13 мульти-таймфрейм). |
| `Dataset *data_alloc_dataset(...)` | Выделение памяти, заполнение features. |
| `void data_normalize_features(Dataset *ds)` | Нормализация фичей в [0,1]. |
| `DatasetSplit data_split_train_test(Dataset *ds, double train_ratio)` | Хронологическое разделение (shallow copy). |
| `void data_free_dataset(Dataset *ds)` | Освобождение всех ресурсов. |
| `void data_free_split(DatasetSplit *split)` | Освобождение структур split. |

#### `include/model.h` / `src/model.c`

| Функция | Описание |
|---------|----------|
| `int model_save(...)` | Сохранение: метаданные, `[normalization]`, `[lstm_weights]`, `[output_weights]` (num_outputs строк × num_cells весов + bias array). |
| `int model_load(const char *filepath, LSTMNetwork *net)` | Загрузка модели. |

### Тесты

31 тест, разбиты по файлам, все проходят:

**Общие LSTM тесты (13):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 1 | `test_forward.c` | `test_forward_pass` | Forward pass без NaN/Inf |
| 2 | `test_forward.c` | `test_concatenated_storage` | concatenated хранит все n+f элементов |
| 3 | `test_forward.c` | `test_state_propagation` | H и C различаются между шагами |
| 4 | `test_backward.c` | `test_backward_pass` | Градиенты ненулевые после BPTT |
| 5 | `test_backward.c` | `test_last_timestep_grad` | Градиенты на последнем шаге ненулевые |
| 6 | `test_backward.c` | `test_output_grad_accumulation` | Аккумуляция градиентов output layer |
| 7 | `test_update.c` | `test_update` | Веса изменяются после update |
| 8 | `test_update.c` | `test_batch_normalization` | batch_size нормализация |
| 9 | `test_model_io.c` | `test_save_load` | Save/Load сохраняет предсказания |
| 10 | `test_deterministic.c` | `test_deterministic_forward_64` | 64 cells × 64 steps: H, C совпадают с аналитикой |
| 11 | `test_gradient.c` | `test_bptt_gradient_check_64` | 64 cells × 5 steps: 8 gradient checks |
| 12 | `test_deterministic.c` | `test_deterministic_forward_128` | 128 cells × 256 steps: H, C совпадают с аналитикой |
| 13 | `test_gradient.c` | `test_bptt_gradient_check_128` | 128 cells × 4 steps: 10 gradient checks |

**26-признаковые тесты (4):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 14 | `test_features_26.c` | `test_forward_26_features` | Forward pass с 26 признаками без NaN/Inf |
| 15 | `test_features_26.c` | `test_backward_26_features` | BPTT с 26 признаками, градиенты ненулевые |
| 16 | `test_deterministic.c` | `test_deterministic_forward_26` | 32 cells × 8 steps × 26 features: H, C совпадают с аналитикой |
| 17 | `test_gradient.c` | `test_bptt_gradient_check_26` | 32 cells × 8 steps × 26 features: 8 gradient checks |

**Data I/O тесты (7):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 18 | `test_data_io.c` | `test_load_raw_basic` | Загрузка CSV: >1000 строк, валидные OHLC |
| 19 | `test_data_io.c` | `test_load_raw_chronological` | Строки в хронологическом порядке |
| 20 | `test_data_io.c` | `test_aggregate_h1_identity` | h=1 возвращает исходные данные |
| 21 | `test_data_io.c` | `test_aggregate_h2` | h=2: Open/High/Low/Close/TickVol/Date проверены |
| 22 | `test_data_io.c` | `test_aggregate_h4` | h=4: аналогично |
| 23 | `test_data_io.c` | `test_aggregate_h8` | h=8: аналогично |
| 24 | `test_data_io.c` | `test_aggregate_double_h2` | h=2 затем h=2 равно h=4 |

**OHLC тесты (3):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 25 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE уменьшается (OHLC labels, many-to-many) |
| 26 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels в [-1,1] |
| 27 | `ohlc/test_deterministic_ohlc.c` | `test_deterministic_ohlc` | Forward+BPTT vs scalar (1 cell, 2 шага, linear) |

**Direction тесты (4):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 28 | `direction/test_data_direction.c` | `test_direction_labels` | Все labels ровно 0 или 1 |
| 29 | `direction/test_training_direction.c` | `test_training_loop_direction` | BCE уменьшается (sigmoid output, labels 0/1) |
| 30 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks (sigmoid+BCE), max rel_err ~3e-7 |
| 31 | `direction/test_deterministic_direction.c` | `test_deterministic_direction` | Forward+BPTT vs scalar (1 cell, 2 шага, sigmoid+BCE) |


## 🇬🇧

This project implements an LSTM (Long Short-Term Memory) neural network in C90 standard for forecasting EUR/USD quotes. It operates in two modes: **OHLC** (predicting Open/High/Low/Close prices) and **Direction** (predicting price direction up/down).

### Prerequisites

- C compiler with C90 support (gcc, clang, MSVC, etc.)
- CMake 3.10+
- math library (libm, linked automatically)

### Build

```bash
# Main binary + tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Tests only
make test_lstm

# Main binary only
make LSTM_C90
```

### Running Tests

```bash
./test_lstm
# Expected output: Results: 31/31 tests passed
```

### Data Format

CSV file in MetaTrader export format (tab-delimited, double-quoted fields):

```
<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
2009.06.12\t00:00:00\t1.40954\t1.41129\t1.40929\t1.41069\t1002\t0\t20
```

- **Timeframe:** H1 (hourly bars)
- **Data range:** 2009-06-12 … 2025-09-25
- **Used columns:** OPEN, HIGH, LOW, CLOSE, TICKVOL
- **Ignored:** VOL (always 0), SPREAD (not used)
- To replace data — export CSV from MetaTrader in the same format

### Architecture

```
Input (26 features × 64 steps) → LSTM (128 cells) → LINEAR (128→num_outputs) → predict_count candles
           many-to-many: output at every step
```

- **Language standard:** C90 (no C99 features)
- **LSTM cells:** 128 (hidden state size, independent of sequence length)
- **Features:** 26 (13 base + 13 multi-timeframe, per time step)
- **Time steps:** 64 (input sequence length)
- **Outputs:** `predict_count × 4` (OHLC) or `predict_count` (Direction), per step
- **Architecture:** many-to-many — output layer fires at every step
- **Default prediction horizon:** 1 candle (`--output 1`), can be increased via CLI
- **Loss function:** MSE (OHLC) or BCE (Direction)
- **Autoregression:** none (single forward pass → all candles directly)

### Modes

**OHLC** (`--mode ohlc`, default): predicts future OHLC prices as returns.
At each step `t`, the LSTM predicts `predict_count` future OHLC candles. Labels for step `t` are returns of the next candles relative to `close_t`. Outputs: `predict_count × 4` (4 OHLC values per candle).

**Direction** (`--mode direction`): predicts price direction (up/down).
Labels: `1.0` if `close_{t+k} > close_t`, else `0.0` (binary classification). Outputs: `predict_count` (1 direction per step). Output activation: **sigmoid** (output in (0, 1)). Loss: **BCE** (Binary Cross-Entropy). Threshold: `pred > 0.5` → up, else → down.

### Training (many-to-many)

Loss sums over all `seq_len × num_outputs` outputs, giving more gradient signal per sample. Optimizer: mini-batch SGD + momentum, per-element gradient clipping to [-5, +5].

**Ctrl+C** during training — model is saved gracefully (signal handler).

### Prediction

Single forward pass → take outputs from the **last step** (`outputs[(seq_len-1) × num_outputs + o]`) → denormalize → predicted values.
- **No autoregression:** single forward pass → all candles directly

### Features (26)

**13 base features** (indices 0–12):

| # | Name | Description |
|---|------|-------------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1}, clipped [-10,10] |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 bars) |
| 6 | `rsi` | RSI(14) / 100, [0,1] |
| 7 | `macd_hist` | (EMA12 - EMA26) - EMA9(signal) |
| 8 | `bb_position` | (close - BB_lower) / (BB_upper - BB_lower), BB = MA20 ± 2σ |
| 9 | `atr_norm` | ATR(14) / close |
| 10 | `stoch_k` | Stochastic %K(14) / 100, [0,1] |
| 11 | `hour` | hour_of_day / 23 |
| 12 | `dow` | day_of_week / 6 |

**13 multi-timeframe features** (indices 13–25): same 13 features computed on H2-aggregated data (every 2 H1 candles → 1 H2 candle), overlaid onto H1 rows.

| # | Name | Description |
|---|------|-------------|
| 13 | `mtf_candle_return` | candle_return on H2 |
| 14 | `mtf_candle_range` | candle_range on H2 |
| 15 | `mtf_close_return` | close_return on H2 |
| 16 | `mtf_volume_return` | volume_return on H2 |
| 17 | `mtf_ma_diff` | ma_diff on H2 |
| 18 | `mtf_volatility` | volatility on H2 |
| 19 | `mtf_rsi` | rsi on H2 |
| 20 | `mtf_macd_hist` | macd_hist on H2 |
| 21 | `mtf_bb_position` | bb_position on H2 |
| 22 | `mtf_atr_norm` | atr_norm on H2 |
| 23 | `mtf_stoch_k` | stoch_k on H2 |
| 24 | `mtf_hour` | hour on H2 |
| 25 | `mtf_dow` | dow on H2 |

### Outputs (OHLC mode)

For each of `predict_count` future candles (k=1..predict_count), 4 returns relative to `ref_close` (last close in sequence):

| Index | Output | Formula |
|--------|-------|---------|
| k·4+0 | open_return_k | (open_{t+k} - ref_close) / ref_close |
| k·4+1 | high_return_k | (high_{t+k} - ref_close) / ref_close |
| k·4+2 | low_return_k | (low_{t+k} - ref_close) / ref_close |
| k·4+3 | close_return_k | (close_{t+k} - ref_close) / ref_close |

Denormalization: `price = ref_close × (1 + denorm(pred))`

### Outputs (Direction mode)

For each of `predict_count` future steps (k=1..predict_count): 1 value — probability of upward direction.

| Index | Output | Formula |
|--------|-------|---------|
| k | direction_k | 1.0 if close_{t+k} > close_t, else 0.0 |

Threshold: `pred > 0.5` → up (1), else → down (0).

### Timeframe Aggregation (`--h`)

The `--h N` parameter groups every N consecutive H1 candles into one, creating a higher timeframe. All computations (features, labels, training, prediction) operate on aggregated data.

Aggregation rules for N H1 candles into one:

| Field | Value |
|-------|-------|
| Open | open of first candle in group |
| High | max(high) across all candles in group |
| Low | min(low) across all candles in group |
| Close | close of last candle in group |
| TickVol | sum(tickvol) across all candles |
| Date/Time | first candle's date/time |

Incomplete last group is dropped.

| Parameter | Data rows | Description |
|-----------|-----------|-------------|
| `--h 1` (default) | ~101,000 | original H1 candles |
| `--h 2` | ~50,500 | H2 — aggregate 2 candles |
| `--h 4` | ~25,250 | H4 — aggregate 4 candles |
| `--h 8` | ~12,600 | H8 — aggregate 8 candles |

### Parameters

**CLI-regulated:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cells` | 128 | LSTM cells (hidden state size) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Input sequence length (bars) |
| `--mode` | `ohlc` | Mode: `ohlc`, `direction`, `ohlcpredict`, `directionpredict`, `realdata` |
| `--output` | 1 | Number of outputs (OHLC vectors or direction steps) |
| `--datetime` | — | `YYYY.MM.DDThh:mm:ss` — datetime for prediction (required for predict modes) |
| `--from` | — | `YYYY.MM.DDThh:mm:ss` — range start for `--mode realdata` |
| `--to` | — | `YYYY.MM.DDThh:mm:ss` — range end for `--mode realdata` |
| `--data` | `../resources/EURUSD_*.csv` | Path to CSV file |
| `--model` | `../resources/model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt` | Path to model file |
| `--h` | 1 | Timeframe aggregation: group N H1 candles into one (H1=1, H2=2, H4=4, ...) |

**Hardcoded** (in `lstm.h` / `main.c`):

| Parameter | Value | Where |
|-----------|-------|-------|
| `OUTPUTS_PER_CANDLE` | 4 (OHLC) | `lstm.h` |
| `DEFAULT_OHLC_PREDICT_COUNT` | 1 | `lstm.h` |
| `PREDICTION_THRESHOLD` | 0.5 | `lstm.h` |
| Batch size | 32 | `main.c` |
| Momentum (SGD) | 0.9 | `main.c` |
| Seed | 42 | `main.c` |
| Weight decay (L2) | 1e-6 | `lstm.c` |
| Gradient clipping | [-5.0, +5.0] | `utils.h` |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) | `lstm.c` |
| Forget gate bias init | 1.0 (Jozefowicz et al., 2015) | `lstm.c` |
| Train/test split | 80/20 chronological | `data.c` |
| `NUM_RAW_FEATURES` | 26 | `data.h` |
| `NUM_BASE_FEATURES` | 13 | `data.h` |
| `NUM_MTF_FEATURES` | 13 | `data.h` |

### Modes

**1. Training** (`--mode ohlc` or `--mode direction`):
```bash
./LSTM_C90 --mode ohlc --cells 128 --epochs 100 --lr 0.005 --seq 64 --output 1
./LSTM_C90 --mode ohlc --cells 64 --epochs 50 --lr 0.001 --seq 32 --output 5
./LSTM_C90 --mode direction --epochs 50 --lr 0.01 --output 3
./LSTM_C90 --mode direction --epochs 50 --lr 0.01 --output 3 --h 4
```
Model saved to `model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt`. **Ctrl+C** — graceful model save.

**2. Prediction** (`--mode ohlcpredict` or `--mode directionpredict`):
```bash
./LSTM_C90 --mode ohlcpredict --datetime 2024.01.15T08:00:00 --output 5
./LSTM_C90 --mode directionpredict --datetime 2024.01.15T08:00:00 --output 3
./LSTM_C90 --mode ohlcpredict --datetime 2024.01.15T08:00:00 --output 5 --h 4
./LSTM_C90 --mode directionpredict --datetime 2024.01.15T08:00:00 --output 3 --h 4
```
If `predict_count`, `mode`, `seq` or `--h` doesn't match the model — error and exit. Single forward pass → `num_outputs` outputs → denormalize → predicted values. Only prediction rows are output, in the quantity specified by `--output`. **No autoregression.**

**3. Real data** (`--mode realdata`):
```bash
# All quotes from file
./LSTM_C90 --mode realdata
# From specified datetime to end of file
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00
# In range from --from to --to
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00 --to 2024.02.15T08:00:00
# With H4 timeframe aggregation
./LSTM_C90 --mode realdata --from 2024.01.15T08:00:00 --to 2024.02.15T08:00:00 --h 4
```
Output: `YYYY.MM.DDThh:mm:ss O H L C TICKVOL` — one line per candle. Aggregation via `--h` is applied the same way as in training and prediction.

### Model File Format

Human-readable text format:

```
# LSTM Model - Text Format
# num_cells: 128
# num_features: 26
# num_outputs: 4
# predict_count: 1
# mode: ohlc
# output_activation: 0
# seq_len: 64
# epochs_trained: 100
# best_train_mse: ...
# best_test_direction_accuracy: ...

[normalization]
feature_name: candle_return | min: ... | max: ...
...
label_min: ... ...
label_max: ... ...

[lstm_weights]
# Gate: forget
forget_recurrent:
  row[0]: w0 w1 ... w_combined
...
forget_input:
  row[0]: ...
forget_bias: b0 b1 ... b_n

[output_weights]
weight_output_layer:
  row[0]: w0 w1 ... w_n
bias_output_layer: b0 b1 ... b_num_out
```

Weights are saved with full double precision (`%.17g`). `output_activation`: 0=linear (OHLC), 1=sigmoid (Direction).

### Project structure

```
├── CMakeLists.txt               # CMake configuration (C90)
├── main.c                       # Entry point, CLI, training loop
├── include/
│   ├── types.h, utils.h, data.h, lstm.h, model.h   # Common
│   ├── ohlc/
│   │   ├── data_ohlc.h, predict_ohlc.h
│   └── direction/
│       ├── data_direction.h, predict_direction.h
├── src/
│   ├── utils.c, data.c, lstm.c, model.c   # Common
│   ├── ohlc/
│   │   ├── data_ohlc.c, predict_ohlc.c
│   └── direction/
│       ├── data_direction.c, predict_direction.c
├── tests/
│   ├── test_utils.h/.c, test_main.c       # Shared helpers, runner
│   ├── test_forward.c, test_backward.c, test_update.c   # Common LSTM tests
│   ├── test_model_io.c, test_deterministic.c, test_gradient.c
│   ├── test_features_26.c                 # 26-feature tests
│   ├── ohlc/
│   │   ├── test_training_ohlc.c, test_data_ohlc.c
│   │   └── test_deterministic_ohlc.c
│   └── direction/
│       ├── test_data_direction.c, test_training_direction.c
│       ├── test_sigmoid_gradient.c
│       └── test_deterministic_direction.c
└── resources/
    ├── EURUSD_H1_*.csv
    └── model_{pair}_{mode}_cells_{N}_seq_{N}_count_{N}.txt
```

### Tests

31 tests, split across files, all passing:

**Common LSTM tests (13):**

| # | File | Test | Description |
|---|------|------|-------------|
| 1 | `test_forward.c` | `test_forward_pass` | Forward pass produces no NaN/Inf |
| 2 | `test_forward.c` | `test_concatenated_storage` | concatenated stores all n+f elements |
| 3 | `test_forward.c` | `test_state_propagation` | H and C differ between timesteps |
| 4 | `test_backward.c` | `test_backward_pass` |: Gradients are non-zero after BPTT |
| 5 | `test_backward.c` | `test_last_timestep_grad` | Gradients at last timestep are non-zero |
| 6 | `test_backward.c` | `test_output_grad_accumulation` | Output layer gradient accumulation |
| 7 | `test_update.c` | `test_update` | Weights change after update |
| 8 | `test_update.c` | `test_batch_normalization` | batch_size normalization |
| 9 | `test_model_io.c` | `test_save_load` | Save/Load preserves predictions |
| 10 | `test_deterministic.c` | `test_deterministic_forward_64` | 64 cells × 64 steps: H, C match analytical |
| 11 | `test_gradient.c` | `test_bptt_gradient_check_64` | 64 cells × 5 steps: 8 gradient checks |
| 12 | `test_deterministic.c` | `test_deterministic_forward_128` | 128 cells × 256 steps: H, C match analytical |
| 13 | `test_gradient.c` | `test_bptt_gradient_check_128` | 128 cells × 4 steps: 10 gradient checks |

**26-feature tests (4):**

| # | File | Test | Description |
|---|------|------|-------------|
| 14 | `test_features_26.c` | `test_forward_26_features` | Forward pass with 26 features, no NaN/Inf |
| 15 | `test_features_26.c` | `test_backward_26_features` | BPTT with 26 features, gradients non-zero |
| 16 | `test_deterministic.c` | `test_deterministic_forward_26` | 32 cells × 8 steps × 26 features: H, C match analytical |
| 17 | `test_gradient.c` | `test_bptt_gradient_check_26` | 32 cells × 8 steps × 26 features: 8 gradient checks |

**Data I/O tests (7):**

| # | File | Test | Description |
|---|------|------|-------------|
| 18 | `test_data_io.c` | `test_load_raw_basic` | CSV loading: >1000 rows, valid OHLC |
| 19 | `test_data_io.c` | `test_load_raw_chronological` | Rows in chronological order |
| 20 | `test_data_io.c` | `test_aggregate_h1_identity` | h=1 returns original data unchanged |
| 21 | `test_data_io.c` | `test_aggregate_h2` | h=2: Open/High/Low/Close/TickVol/Date verified |
| 22 | `test_data_io.c` | `test_aggregate_h4` | h=4: same verification |
| 23 | `test_data_io.c` | `test_aggregate_h8` | h=8: same verification |
| 24 | `test_data_io.c` | `test_aggregate_double_h2` | h=2 then h=2 equals h=4 |

**OHLC tests (3):**

| # | File | Test | Description |
|---|------|------|-------------|
| 25 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE decreases (OHLC labels, many-to-many) |
| 26 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels in [-1,1] |
| 27 | `ohlc/test_deterministic_ohlc.c` | `test_deterministic_ohlc` | Forward+BPTT vs scalar (1 cell, 2 steps, linear) |

**Direction tests (4):**

| # | File | Test | Description |
|---|------|------|-------------|
| 28 | `direction/test_data_direction.c` | `test_direction_labels` | All labels are exactly 0 or 1 |
| 29 | `direction/test_training_direction.c` | `test_training_loop_direction` | BCE decreases (sigmoid output, labels 0/1) |
| 30 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks (sigmoid+BCE), max rel_err ~3e-7 |
| 31 | `direction/test_deterministic_direction.c` | `test_deterministic_direction` | Forward+BPTT vs scalar (1 cell, 2 steps, sigmoid+BCE) |
