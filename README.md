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
# Ожидаемый результат: Results: 20/20 tests passed
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
Вход (11 признаков × 64 шага) → LSTM (128 ячеек) → LINEAR (128→num_outputs) → predict_count свечей
           many-to-many: output на каждом шаге
```

- **Стандарт языка:** C90 (без C99-фич)
- **LSTM ячеек:** 128 (размер hidden state, не зависит от длины последовательности)
- **Признаков:** 11 (на каждый временной шаг)
- **Временных шагов:** 64 (длина входной последовательности)
- **Выходов:** `predict_count × 4` (OHLC) или `predict_count` (Direction), на каждый шаг
- **Архитектура:** many-to-many — output layer срабатывает на каждом шаге
- **Горизонт прогноза по умолчанию:** 1 свеча (`--ohlcpredictcount 1`), можно изменить до 5+ через CLI
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

### Признаки (11)

| # | Имя | Описание |
|---|-----|----------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 баров) |
| 6 | `rsi` | RSI(14) / 100, [0,1] |
| 7 | `macd_hist` | (EMA12 - EMA26) - EMA9(сигнал) |
| 8 | `bb_position` | (close - BB_lower) / (BB_upper - BB_lower), BB = MA20 ± 2σ |
| 9 | `atr_norm` | ATR(14) / close |
| 10 | `stoch_k` | Stochastic %K(14) / 100, [0,1] |

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

### Параметры

**CLI-регулируемые:**

| Флаг | Дефолт | Описание |
|------|--------|----------|
| `--cells` | 128 | Кол-во LSTM ячеек (размер hidden state) |
| `--epochs` | 100 | Кол-во эпох обучения |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Длина входной последовательности (баров) |
| `--mode` | `ohlc` | Режим: `ohlc` (прогноз OHLC) или `direction` (прогноз направления 0/1) |
| `--ohlcpredictcount` | 1 | Кол-во OHLC векторов (только для `--mode ohlc`) |
| `--directioncount` | 1 | Кол-во шагов прогноза направления (только для `--mode direction`) |
| `--data` | `../resources/EURUSD_*.csv` | Путь к CSV-файлу |
| `--model` | `../resources/model_{mode}_predic_count_N.txt` | Путь к файлу модели |
| `--line` | 0 | Номер строки CSV для прогноза (0 = обучение) |
| `--real` | 0 | Вывод реальных данных M баров начиная со строки N (с `--line`) |

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
| `NUM_RAW_FEATURES` | 11 | `data.h` |

### Режимы работы

**1. Обучение** (по умолчанию):
```bash
./LSTM_C90
./LSTM_C90 --cells 64 --epochs 50 --lr 0.001 --seq 32 --ohlcpredictcount 5
./LSTM_C90 --mode direction --epochs 50 --lr 0.01
```
Модель сохраняется в `model_{mode}_predic_count_N.txt` (N = predict_count). **Ctrl+C** — корректное сохранение модели.

**2. Прогноз** (`--line N` без `--real`):
```bash
./LSTM_C90 --line 100000 --ohlcpredictcount 5
./LSTM_C90 --mode direction --line 100000 --directioncount 5
```
Если `predict_count` или `mode` не совпадают с моделью — ошибка и выход. Один forward pass → `num_outputs` выходов → денормализация → предсказанные значения. **Без авторегрессии.**

**3. Реальные данные** (`--line N --real M`):
```bash
./LSTM_C90 --line 100000 --real 64
```
Вывод OHLC для M баров начиная со строки N.

### Скрипты (PowerShell)

| Скрипт | Описание |
|--------|----------|
| `scripts/build.ps1` | Сборка основного бинарника через CMake + MinGW. Использует встроенный в CLion CMake — замените путь при необходимости. |
| `scripts/train.ps1` | Запуск обучения. Параметры: `-Mode` (Quick/Medium/Full/OneEpoch), `-Epochs`, `-DataPercent`. **Внимание:** переопределяет `--lr` на 0.0005 (не 0.005). |
| `scripts/predict.ps1` | Запуск прогноза. Параметры: `-Line`, `-Count`. **Внимание:** использует флаг `--predict`, которого нет в main.c — замените на `--ohlcpredictcount`. |

```powershell
./scripts/build.ps1
./scripts/train.ps1 -Mode Quick      # 10 эпох, 10% данных
./scripts/train.ps1 -Mode Full       # 100 эпох, 100% данных
./scripts/predict.ps1 -Line 100000 -Count 5
```

### Формат файла модели

Текстовый формат, читаемый человеком:

```
# LSTM Model - Text Format
# num_cells: 128
# num_features: 11
# num_outputs: 4
# predict_count: 1
# mode: ohlc
# output_activation: 0
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
│   ├── ohlc/
│   │   ├── test_training_ohlc.c     # Training loop (OHLC labels)
│   │   ├── test_data_ohlc.c        # Volume clip & label norm
│   │   └── test_deterministic_ohlc.c # Forward+BPTT vs scalar (linear)
│   └── direction/
│       ├── test_data_direction.c   # Labels 0/1
│       ├── test_training_direction.c # Training loop (sigmoid+BCE)
│       ├── test_sigmoid_gradient.c  # Gradient check (sigmoid+BCE)
│       └── test_deterministic_direction.c # Forward+BPTT vs scalar (sigmoid+BCE)
├── resources/
│   ├── EURUSD_H1_*.csv
│   └── model_{mode}_predic_count_N.txt
└── scripts/
    ├── build.ps1, train.ps1, predict.ps1
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
| `double **data_compute_raw_features(...)` | Feature engineering: 11 признаков на бар (6 базовых + 5 технических индикаторов). |
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

20 тестов, разбиты по файлам, все проходят:

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

**OHLC тесты (3):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 14 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE уменьшается (OHLC labels, many-to-many) |
| 15 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels в [-1,1] |
| 16 | `ohlc/test_deterministic_ohlc.c` | `test_deterministic_ohlc` | Forward+BPTT vs scalar (1 cell, 2 шага, linear) |

**Direction тесты (4):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 17 | `direction/test_data_direction.c` | `test_direction_labels` | Все labels ровно 0 или 1 |
| 18 | `direction/test_training_direction.c` | `test_training_loop_direction` | BCE уменьшается (sigmoid output, labels 0/1) |
| 19 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks (sigmoid+BCE), max rel_err ~3e-7 |
| 20 | `direction/test_deterministic_direction.c` | `test_deterministic_direction` | Forward+BPTT vs scalar (1 cell, 2 шага, sigmoid+BCE) |


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
# Expected output: Results: 20/20 tests passed
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
Input (11 features × 64 steps) → LSTM (128 cells) → LINEAR (128→num_outputs) → predict_count candles
           many-to-many: output at every step
```

- **Language standard:** C90 (no C99 features)
- **LSTM cells:** 128 (hidden state size, independent of sequence length)
- **Features:** 11 (per time step)
- **Time steps:** 64 (input sequence length)
- **Outputs:** `predict_count × 4` (OHLC) or `predict_count` (Direction), per step
- **Architecture:** many-to-many — output layer fires at every step
- **Default prediction horizon:** 1 candle (`--ohlcpredictcount 1`), can be increased via CLI
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

### Features (11)

| # | Name | Description |
|---|------|-------------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 bars) |
| 6 | `rsi` | RSI(14) / 100, [0,1] |
| 7 | `macd_hist` | (EMA12 - EMA26) - EMA9(signal) |
| 8 | `bb_position` | (close - BB_lower) / (BB_upper - BB_lower), BB = MA20 ± 2σ |
| 9 | `atr_norm` | ATR(14) / close |
| 10 | `stoch_k` | Stochastic %K(14) / 100, [0,1] |

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

### Parameters

**CLI-regulated:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cells` | 128 | LSTM cells (hidden state size) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Input sequence length (bars) |
| `--mode` | `ohlc` | Mode: `ohlc` (OHLC forecast) or `direction` (0/1 direction forecast) |
| `--ohlcpredictcount` | 1 | Number of OHLC vectors (only for `--mode ohlc`) |
| `--directioncount` | 1 | Number of direction steps (only for `--mode direction`) |
| `--data` | `../resources/EURUSD_*.csv` | Path to CSV file |
| `--model` | `../resources/model_{mode}_predic_count_N.txt` | Path to model file |
| `--line` | 0 | CSV row number for prediction (0 = training mode) |
| `--real` | 0 | Output real data for M bars starting at row N (with `--line`) |

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
| `NUM_RAW_FEATURES` | 11 | `data.h` |

### Modes

**1. Training** (default):
```bash
./LSTM_C90
./LSTM_C90 --cells 64 --epochs 50 --lr 0.001 --seq 32 --ohlcpredictcount 5
./LSTM_C90 --mode direction --epochs 50 --lr 0.01
```
Model saved to `model_{mode}_predic_count_N.txt` (N = predict_count). **Ctrl+C** — graceful model save.

**2. Prediction** (`--line N` without `--real`):
```bash
./LSTM_C90 --line 100000 --ohlcpredictcount 5
./LSTM_C90 --mode direction --line 100000 --directioncount 5
```
If `predict_count` or `mode` doesn't match the model — error and exit. Single forward pass → `num_outputs` outputs → denormalize → predicted values. **No autoregression.**

**3. Real data** (`--line N --real M`):
```bash
./LSTM_C90 --line 100000 --real 64
```
Output OHLC for M bars starting at row N.

### Scripts (PowerShell)

| Script | Description |
|--------|-------------|
| `scripts/build.ps1` | Builds main binary via CMake + MinGW. Uses CLion-bundled CMake — replace path if needed. |
| `scripts/train.ps1` | Runs training. Params: `-Mode` (Quick/Medium/Full/OneEpoch), `-Epochs`, `-DataPercent`. **Note:** overrides `--lr` to 0.0005 (not 0.005). |
| `scripts/predict.ps1` | Runs prediction. Params: `-Line`, `-Count`. **Note:** uses `--predict` flag which doesn't exist in main.c — replace with `--ohlcpredictcount`. |

```powershell
./scripts/build.ps1
./scripts/train.ps1 -Mode Quick      # 10 epochs, 10% data
./scripts/train.ps1 -Mode Full       # 100 epochs, 100% data
./scripts/predict.ps1 -Line 100000 -Count 5
```

### Model File Format

Human-readable text format:

```
# LSTM Model - Text Format
# num_cells: 128
# num_features: 11
# num_outputs: 4
# predict_count: 1
# mode: ohlc
# output_activation: 0
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
│   ├── ohlc/
│   │   ├── test_training_ohlc.c, test_data_ohlc.c
│   │   └── test_deterministic_ohlc.c
│   └── direction/
│       ├── test_data_direction.c, test_training_direction.c
│       ├── test_sigmoid_gradient.c
│       └── test_deterministic_direction.c
├── resources/
│   ├── EURUSD_H1_*.csv
│   └── model_{mode}_predic_count_N.txt
└── scripts/
    ├── build.ps1, train.ps1, predict.ps1
```

### Tests

20 tests, split across files, all passing:

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

**OHLC tests (3):**

| # | File | Test | Description |
|---|------|------|-------------|
| 14 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE decreases (OHLC labels, many-to-many) |
| 15 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels in [-1,1] |
| 16 | `ohlc/test_deterministic_ohlc.c` | `test_deterministic_ohlc` | Forward+BPTT vs scalar (1 cell, 2 steps, linear) |

**Direction tests (4):**

| # | File | Test | Description |
|---|------|------|-------------|
| 17 | `direction/test_data_direction.c` | `test_direction_labels` | All labels are exactly 0 or 1 |
| 18 | `direction/test_training_direction.c` | `test_training_loop_direction` | BCE decreases (sigmoid output, labels 0/1) |
| 19 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks (sigmoid+BCE), max rel_err ~3e-7 |
| 20 | `direction/test_deterministic_direction.c` | `test_deterministic_direction` | Forward+BPTT vs scalar (1 cell, 2 steps, sigmoid+BCE) |
