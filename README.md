# LSTM

---

## 🇷🇺

Проект реализует LSTM (Long Short-Term Memory) нейронную сеть на стандарте C90 для прогнозирования OHLC котировок EUR/USD.

### Архитектура

```
Вход (6 признаков × 64 шага) → LSTM (128 ячеек) → LINEAR (128→20) → 5 свечей OHLC
           many-to-many: output на каждом шаге
```

- **Стандарт языка:** C90 (без C99-фич)
- **LSTM ячеек:** 128 (размер hidden state, не зависит от длины последовательности)
- **Признаков:** 6 (на каждый временной шаг)
- **Временных шагов:** 64 (длина входной последовательности)
- **Выходов:** 20 на каждый шаг (5 свечей × OHLC)
- **Архитектура:** many-to-many — output layer срабатывает на каждом шаге
- **Горизонт прогноза:** 5 свечей
- **Функция потерь:** MSE (`0.5 × Σ_t Σ_o (pred_t,o - target_t,o)²` по всем шагам и выходам)
- **Авторегрессия:** нет (один forward pass → сразу 5 свечей)

### Обучение (many-to-many)

На каждом шаге `t` LSTM предсказывает 5 будущих свечей OHLC. Метки для шага `t` — returns следующих 5 свечей относительно `close_t`. Loss суммируется по всем `seq_len × 20` выходам, что даёт больше градиентного сигнала на каждый семпл.

### Предсказание

Один forward pass → берём выходы **последнего шага** (`outputs[(seq_len-1) × 20 + o]`) → денормализация → 5 OHLC свечей.
- **Функция потерь:** MSE (`0.5 × Σ(pred_o - target_o)²` по всем 20 выходам)
- **Авторегрессия:** нет (один forward pass → сразу 5 свечей)

### Признаки (6, без изменений)

| # | Имя | Описание |
|---|-----|----------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 баров) |

### Выходы (20)

Для каждой из 5 будущих свечей (k=1..5) предсказываются 4 returns относительно `ref_close` (close последнего бара последовательности):

| Индекс | Выход | Формула |
|--------|-------|---------|
| k·4+0 | open_return_k | (open_{t+k} - ref_close) / ref_close |
| k·4+1 | high_return_k | (high_{t+k} - ref_close) / ref_close |
| k·4+2 | low_return_k | (low_{t+k} - ref_close) / ref_close |
| k·4+3 | close_return_k | (close_{t+k} - ref_close) / ref_close |

Денормализация: `price = ref_close × (1 + denorm(pred))`

### Параметры

**CLI-регулируемые:**

| Флаг | Дефолт | Описание |
|------|--------|----------|
| `--cells` | 128 | Кол-во LSTM ячеек (размер hidden state) |
| `--epochs` | 100 | Кол-во эпох обучения |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Длина входной последовательности (баров) |
| `--mode` | `ohlc` | Режим: `ohlc` (прогноз OHLC) или `direction` (прогноз направления 1/-1) |
| `--ohlcpredictcount` | 1 | Кол-во OHLC векторов (только для `--mode ohlc`) |
| `--directioncount` | 1 | Кол-во шагов прогноза направления (только для `--mode direction`) |
| `--data` | `../resources/EURUSD_*.csv` | Путь к CSV-файлу |
| `--model` | `../resources/model_ohlc_predic_count_N.txt` | Путь к файлу модели |

**Зашиты в код** (в `lstm.h` / `main.c`):

| Параметр | Значение | Где |
|----------|----------|-----|
| `PREDICTION_HORIZON` | (`--ohlcpredictcount`) | runtime, default 1 |
| `OUTPUTS_PER_CANDLE` | 4 (OHLC) | `lstm.h` |
| `NUM_OUTPUTS` | `ohlc_predict_count × 4` | runtime |
| Batch size | 32 | `main.c` |
| Momentum (SGD) | 0.9 | `main.c` |
| Seed | 42 | `main.c` |
| Weight decay (L2) | 1e-6 | `lstm.c` |
| Gradient clipping | [-5.0, +5.0] | `utils.h` |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) | `lstm.c` |
| Train/test split | 80/20 chronological | `data.c` |

### Режимы работы

**1. Обучение** (по умолчанию):
```bash
./LSTM_C90
./LSTM_C90 --cells 64 --epochs 50 --lr 0.001 --seq 32 --ohlcpredictcount 5
```
Модель сохраняется в `model_ohlc_predic_count_N.txt` (N = `--ohlcpredictcount`).

**2. Прогноз** (`--line N` без `--real`):
```bash
./LSTM_C90 --line 100000 --ohlcpredictcount 5
```
Если `--ohlcpredictcount` не совпадает с моделью — ошибка и выход. Один forward pass → `num_outputs` выходов → денормализация → OHLC свечей. **Без авторегрессии.**

**3. Реальные данные** (`--line N --real M`):
```bash
./LSTM_C90 --line 100000 --real 64
```
Вывод OHLC для M баров начиная со строки N.

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
│       └── predict_direction.c  # Threshold в 1/-1
├── tests/
│   ├── test_utils.h/.c, test_main.c       # Общие хелперы, runner
│   ├── test_forward.c, test_backward.c, test_update.c   # Общие LSTM тесты
│   ├── test_model_io.c, test_deterministic.c, test_gradient.c
│   ├── ohlc/
│   │   ├── test_training_ohlc.c     # Training loop (OHLC labels)
│   │   └── test_data_ohlc.c        # Volume clip & label norm
│   └── direction/
│       ├── test_data_direction.c   # Labels 0/1
│       ├── test_training_direction.c # Training loop (sigmoid)
│       └── test_sigmoid_gradient.c  # Sigmoid gradient check
├── resources/
│   ├── EURUSD_H1_*.csv
│   └── model_{mode}_predic_count_N.txt
└── scripts/
    ├── build.ps1, train.ps1, predict.ps1
```

### API — описание всех функций

#### `include/lstm.h` / `src/lstm.c`

**Константы:**
- `PREDICTION_HORIZON = 5` — горизонт прогноза (свечей)
- `OUTPUTS_PER_CANDLE = 4` — OHLC
- `NUM_OUTPUTS = 20` — всего выходов

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
| `double lstm_compute_loss(LSTMNetwork *net, Dataset *ds)` | MSE: `Σ 0.5·(pred_o - label_o)² / (N·num_outputs)`. |
| `double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds)` | Direction accuracy: доля `sign(pred) == sign(label)` по всем выходам. |
| `void lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len, double *outputs)` | Инференс: один forward pass, заполняет `outputs`. |

#### `include/data.h` / `src/data.c`

| Функция | Описание |
|---------|----------|
| `int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count)` | Загрузка CSV. |
| `Dataset *data_build_dataset(DataRow *raw, int num_raw, int seq_len, int prediction_horizon)` | Feature engineering + labels. Метки: OHLC returns для `prediction_horizon` будущих свечей. `num_outputs = prediction_horizon × 4`. |
| `void data_normalize(Dataset *ds)` | Нормализация фичей в [0,1], меток — symmetric scaling per-output в [-1,1]. |
| `DatasetSplit data_split_train_test(Dataset *ds, double train_ratio)` | Хронологическое разделение (shallow copy). |
| `void data_free_dataset(Dataset *ds)` | Освобождение всех ресурсов. |
| `void data_free_split(DatasetSplit *split)` | Освобождение структур split. |

#### `include/model.h` / `src/model.c`

| Функция | Описание |
|---------|----------|
| `int model_save(...)` | Сохранение: метаданные, `[normalization]`, `[lstm_weights]`, `[output_weights]` (num_outputs строк × num_cells весов + bias array). |
| `int model_load(const char *filepath, LSTMNetwork *net)` | Загрузка модели. |

### Тесты

18 тестов, разбиты по файлам, все проходят:

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

**OHLC тесты (2):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 14 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE уменьшается (OHLC labels, many-to-many) |
| 15 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels в [-1,1] |

**Direction тесты (3):**

| # | Файл | Тест | Описание |
|---|------|------|----------|
| 16 | `direction/test_data_direction.c` | `test_direction_labels` | Все labels ровно 0 или 1 |
| 17 | `direction/test_training_direction.c` | `test_training_loop_direction` | MSE уменьшается с sigmoid активацией |
| 18 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks с sigmoid, max rel_err ~2e-7 |


## 🇬🇧

This project implements an LSTM (Long Short-Term Memory) neural network in C90 standard for forecasting EUR/USD OHLC quotes.

### Architecture

```
Input (6 features × 64 steps) → LSTM (128 cells) → LINEAR (128→20) → 5 OHLC candles
           many-to-many: output at every step
```

- **Language standard:** C90 (no C99 features)
- **LSTM cells:** 128 (hidden state size, independent of sequence length)
- **Features:** 6 (per time step)
- **Time steps:** 64 (input sequence length)
- **Outputs:** 20 per step (5 candles × OHLC)
- **Architecture:** many-to-many — output layer fires at every step
- **Prediction horizon:** 5 candles
- **Loss function:** MSE (`0.5 × Σ_t Σ_o (pred_t,o - target_t,o)²` over all steps and outputs)
- **Autoregression:** none (single forward pass → 5 candles directly)

### Training (many-to-many)

At each step `t`, the LSTM predicts 5 future OHLC candles. Labels for step `t` are returns of the next 5 candles relative to `close_t`. Loss sums over all `seq_len × 20` outputs, giving more gradient signal per sample.

### Prediction

Single forward pass → take outputs from the **last step** (`outputs[(seq_len-1) × 20 + o]`) → denormalize → 5 OHLC candles.
- **Loss function:** MSE (`0.5 × Σ(pred_o - target_o)²` over all 20 outputs)
- **Autoregression:** none (single forward pass → 5 candles directly)

### Features (6, unchanged)

| # | Name | Description |
|---|------|-------------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 bars) |

### Outputs (20)

For each of 5 future candles (k=1..5), 4 returns relative to `ref_close` (last close in sequence):

| Index | Output | Formula |
|--------|-------|---------|
| k·4+0 | open_return_k | (open_{t+k} - ref_close) / ref_close |
| k·4+1 | high_return_k | (high_{t+k} - ref_close) / ref_close |
| k·4+2 | low_return_k | (low_{t+k} - ref_close) / ref_close |
| k·4+3 | close_return_k | (close_{t+k} - ref_close) / ref_close |

Denormalization: `price = ref_close × (1 + denorm(pred))`

### Parameters

**CLI-regulated:**

| Flag | Default | Description |
|------|---------|-------------|
| `--cells` | 128 | LSTM cells (hidden state size) |
| `--epochs` | 100 | Training epochs |
| `--lr` | 0.005 | Learning rate (SGD) |
| `--seq` | 64 | Input sequence length (bars) |
| `--mode` | `ohlc` | Mode: `ohlc` (OHLC forecast) or `direction` (1/-1 direction forecast) |
| `--ohlcpredictcount` | 1 | Number of OHLC vectors (only for `--mode ohlc`) |
| `--directioncount` | 1 | Number of direction steps (only for `--mode direction`) |
| `--data` | `../resources/EURUSD_*.csv` | Path to CSV file |
| `--model` | `../resources/model_ohlc_predic_count_N.txt` | Path to model file |

**Hardcoded** (in `lstm.h` / `main.c`):

| Parameter | Value | Where |
|-----------|-------|-------|
| `PREDICTION_HORIZON` | (`--ohlcpredictcount`) | runtime, default 1 |
| `OUTPUTS_PER_CANDLE` | 4 (OHLC) | `lstm.h` |
| `NUM_OUTPUTS` | `ohlc_predict_count × 4` | runtime |
| Batch size | 32 | `main.c` |
| Momentum (SGD) | 0.9 | `main.c` |
| Seed | 42 | `main.c` |
| Weight decay (L2) | 1e-6 | `lstm.c` |
| Gradient clipping | [-5.0, +5.0] | `utils.h` |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) | `lstm.c` |
| Train/test split | 80/20 chronological | `data.c` |

### Modes

**1. Training** (default):
```bash
./LSTM_C90
./LSTM_C90 --cells 64 --epochs 50 --lr 0.001 --seq 32 --ohlcpredictcount 5
```
Model saved to `model_ohlc_predic_count_N.txt` (N = `--ohlcpredictcount`).

**2. Prediction** (`--line N` without `--real`):
```bash
./LSTM_C90 --line 100000 --ohlcpredictcount 5
```
If `--ohlcpredictcount` doesn't match the model — error and exit. Single forward pass → `num_outputs` outputs → denormalize → OHLC candles. **No autoregression.**

**3. Real data** (`--line N --real M`):
```bash
./LSTM_C90 --line 100000 --real 64
```
Output OHLC for M bars starting at row N.

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
│   └── direction/
│       ├── test_data_direction.c, test_training_direction.c
│       └── test_sigmoid_gradient.c
├── resources/
│   ├── EURUSD_H1_*.csv
│   └── model_{mode}_predic_count_N.txt
└── scripts/
    ├── build.ps1, train.ps1, predict.ps1
```

### Tests

18 tests, split across files, all passing:

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

**OHLC tests (2):**

| # | File | Test | Description |
|---|------|------|-------------|
| 14 | `ohlc/test_training_ohlc.c` | `test_training_loop_ohlc` | MSE decreases (OHLC labels, many-to-many) |
| 15 | `ohlc/test_data_ohlc.c` | `test_volume_clip_and_label_norm_ohlc` | Volume clipping + labels in [-1,1] |

**Direction tests (3):**

| # | File | Test | Description |
|---|------|------|-------------|
| 16 | `direction/test_data_direction.c` | `test_direction_labels` | All labels are exactly 0 or 1 |
| 17 | `direction/test_training_direction.c` | `test_training_loop_direction` | MSE decreases with sigmoid activation |
| 18 | `direction/test_sigmoid_gradient.c` | `test_sigmoid_gradient_check` | 6 gradient checks with sigmoid, max rel_err ~2e-7 |
