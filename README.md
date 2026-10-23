# LSTM нейросеть на C90 для прогнозирования EUR/USD / LSTM neural network in C90 for EUR/USD forecasting

---

## 🇷🇺 Описание на русском языке

Проект реализует LSTM (Long Short-Term Memory) нейронную сеть на стандарте C90 для прогнозирования финансовых временных рядов (валютная пара EUR/USD).

### Архитектура

```
Вход (6 признаков × 64 шага) → LSTM (128 ячеек) → LINEAR (128→1) → Прогноз
```

- **Стандарт языка:** C90 (без C99-фич)
- **LSTM ячеек:** 128
- **Признаков:** 6
- **Временных шагов:** 64
- **Выход:** регрессия (нормализованный close_return)
- **Функция потерь:** MSE (0.5 × (pred - target)²)

### Признаки (6)

| # | Имя | Описание |
|---|-----|----------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 баров) |

### Параметры обучения

| Параметр | Значение |
|----------|----------|
| Learning rate | 0.0005 |
| Sequence length | 64 |
| Batch size | 32 |
| Epochs | 100 |
| Momentum (SGD) | 0.9 |
| Weight decay (L2) | 1e-6 |
| Gradient clipping | [-5.0, +5.0] |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) |

### Структура проекта

```
├── CMakeLists.txt           # CMake конфигурация (C90)
├── main.c                   # Точка входа, CLI, цикл обучения
├── include/
│   ├── types.h              # LSTMCellState — состояние ячейки для BPTT
│   ├── utils.h              # Матрицы, математика, random, clipping
│   ├── data.h               # DataRow, Dataset, загрузка/нормализация
│   ├── lstm.h               # LSTMNetwork — архитектура сети
│   └── model.h              # Сохранение/загрузка модели
├── src/
│   ├── utils.c              # Реализация утилит
│   ├── data.c               # Разбор CSV, feature engineering, split
│   ├── lstm.c               # Forward pass, BPTT, update weights
│   └── model.c              # Текстовый формат модели
├── tests/
│   └── test_lstm.c          # 14 тестов (включая gradient check)
├── resources/
│   ├── EURUSD_H1_*.csv      # Данные (101053 бара H1)
│   └── model.txt            # Обученная модель
└── scripts/
    ├── build.ps1            # Сборка через CMake + MinGW
    ├── train.ps1            # Запуск обучения (Quick/Medium/Full)
    └── predict.ps1          # Запуск предсказания
```

### API — описание всех функций

#### `include/utils.h` / `src/utils.c`

**Управление памятью:**

| Функция | Описание |
|---------|----------|
| `void *utils_safe_malloc(size_t n)` | Выделение памяти с проверкой. При failure — `exit(EXIT_FAILURE)`. |
| `void *utils_safe_calloc(size_t n, size_t s)` | Выделение и обнуление памяти с проверкой. |
| `void utils_safe_free(void *ptr)` | Освобождение памяти (обёртка над `free`). |

**Матричные операции** (тип `Matrix { double *data; int rows, cols; }`, row-major):

| Функция | Описание |
|---------|----------|
| `Matrix *matrix_alloc(int rows, int cols)` | Создание матрицы, заполненной нулями. |
| `void matrix_free(Matrix *m)` | Освобождение матрицы. |
| `void matrix_set(Matrix *m, int r, int c, double val)` | Установка элемента `m[r][c] = val`. |
| `double matrix_get(Matrix *m, int r, int c)` | Получение элемента `m[r][c]`. |
| `void matrix_fill(Matrix *m, double val)` | Заполнение всех элементов значением `val`. |
| `void matrix_copy(Matrix *dest, Matrix *src)` | Поэлементное копирование. |
| `void matrix_add(Matrix *dest, Matrix *src)` | Поэлементное сложение `dest += src`. |
| `void matrix_sub(Matrix *dest, Matrix *src)` | Поэлементное вычитание `dest -= src`. |
| `void matrix_scale(Matrix *m, double val)` | Масштабирование `m *= val`. |
| `void matrix_elem_mul(Matrix *dest, Matrix *a, Matrix *b)` | Поэлементное умножение. |
| `void matrix_transpose(Matrix *dest, Matrix *src)` | Транспонирование. |
| `void matrix_mul(Matrix *dest, Matrix *a, Matrix *b)` | Матричное умножение `dest = a × b`. |
| `double matrix_dot_product(Matrix *a, Matrix *b)` | Скалярное произведение. |

**Математические функции:**

| Функция | Описание |
|---------|----------|
| `double utils_sigmoid(double x)` | Numerically stable sigmoid: `1/(1+e^{-x})` при x≥0, `e^x/(1+e^x)` при x<0. |
| `double utils_tanh_act(double x)` | Numerically stable tanh: обрезает \|x\|>20, иначе `(e^{2x}-1)/(e^{2x}+1)`. |
| `double utils_relu(double x)` | ReLU: `max(0, x)`. |
| `double utils_binary_cross_entropy(double prediction, double target)` | Binary cross-entropy loss с epsilon-clip. |

**Генерация случайных чисел:**

| Функция | Описание |
|---------|----------|
| `void utils_srand(unsigned int seed)` | Инициализация генератора. |
| `double utils_rand(void)` | Равномерное распределение [0, 1). |
| `double utils_gaussian_random(double mean, double stddev)` | Нормальное распределение (Box-Muller), обр��зка \|z\|>10. |

**Прочее:**

| Функция | Описание |
|---------|----------|
| `double utils_clip(double val, double min_val, double max_val)` | Обрезка значения в диапазон [min, max]. |
| `void utils_trim(char *str)` | Удаление leading/trailing whitespace из строки. |
| `char *utils_strdup(const char *s)` | Дублирование строки (C90-совместимый). |

#### `include/data.h` / `src/data.c`

**Типы данных:**
- **`DataRow`** — одна запись из CSV: date, time, OHLC, tickvol, vol, spread.
- **`Dataset`** — набор данных: features, labels, параметры нормализации.
- **`DatasetSplit`** — разделение на train/test.

| Функция | Описание |
|---------|----------|
| `int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count)` | Загрузка CSV (MetaTrader формат, tab-delimited). Пропускает заголовок. Возвращает 0 при успехе. |
| `Dataset *data_build_dataset(DataRow *raw, int num_raw, int seq_len)` | Feature engineering: вычисляет 6 признаков для каждого бара, создаёт скользящие окна длиной `seq_len`, метки = close_return следующего бара. Min-max нормализация по каждому признаку. |
| `void data_normalize(Dataset *ds)` | Нормализация признаков в [0, 1] используя feat_min/feat_max. |
| `DatasetSplit data_split_train_test(Dataset *ds, double train_ratio)` | Хронологическое разделение (без перемешивания). Train и test разделяют указатели на features/labels (shallow copy). |
| `void data_free_dataset(Dataset *ds)` | Освобождение всех ресурсов Dataset. |
| `void data_free_split(DatasetSplit *split)` | Освобождение структур split (без освобождения shared данных). |
| `int data_save_normalization(const char *filepath, Dataset *ds)` | Сохранение параметров нормализации в текстовый файл. |
| `int data_load_normalization(const char *filepath, Dataset *ds)` | Загрузка параметров нормализации. |

#### `include/lstm.h` / `src/lstm.c`

**Тип `LSTMNetwork`** — содержит для каждого gate (forget, input, output, candidate):
- `weight_*_recurrent` — матрица [n × (n+f)] (recurrent + input веса)
- `weight_*_input` — матрица [n × f] (зарезервировано)
- `bias_*` — вектор [n]
- `dweight_*`, `dbias_*` — градиенты
- `mweight_*`, `mbias_*` — momentum

Плюс output layer: `weight_output_layer` [1 × n], `bias_output_layer`.

| Функция | Описание |
|---------|----------|
| `LSTMNetwork *lstm_init(int num_cells, int num_features, unsigned int seed)` | Создание сети с Xavier/Glorot инициализацией. scale = sqrt(2/(fan_in + num_cells)). Все градиенты и momentum обнулёны. |
| `void lstm_free(LSTMNetwork *net)` | Освобождение всех ресурсов сети. |
| `double lstm_forward(LSTMNetwork *net, const double *sequence, LSTMCellState *cell_states, int seq_len)` | **Прямое распространение.** Для каждого шага t: вычисляет forget/input/output/candidate gates, обновляет cell state `c_t = f_t·c_{t-1} + i_t·g_t`, hidden state `h_t = o_t·tanh(c_t)`. Сохраняет все промежуточные значения в `cell_states[t]` для BPTT. Возвращает `bias_out + Σ W_out[i]·h[i]` (linear output). |
| `void lstm_backward(LSTMNetwork *net, const double *sequence, const LSTMCellState *cell_states, double prediction, double target, int seq_len)` | **Обратное распространение во времени (BPTT).** Вычисляет градиенты всех весов. Начинает с `dL/dpred = pred - target`, распространяет через output layer, затем идёт от t=seq_len-1 до t=0. Для каждого шага: градиенты по output/forget/input/candidate gates, аккумулирует градиенты весов, распространяет `dhidden` и `dcell` к предыдущему шагу (`dcell = dcell * forget_gate`). |
| `void lstm_zero_gradients(LSTMNetwork *net)` | Обнуление всех градиентов (вызывать перед каждым батчем). |
| `void lstm_update(LSTMNetwork *net, double lr, double momentum, int batch_size)` | **SGD + momentum.** Градиенты нормализуются на batch_size, обрезаются в [-5, +5], затем: `v = momentum·v - lr·(grad + weight_decay·weight)`, `weight += v`. |
| `double lstm_compute_loss(LSTMNetwork *net, Dataset *ds)` | MSE loss по всему датасету: `Σ 0.5·(pred - label)² / N`. |
| `double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds)` | Direction accuracy: доля случаев, где `sign(pred) == sign(label)`. |
| `double lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len)` | Инференс: выделяет cell_states, вызывает `lstm_forward`, освобождает. |
| `void lstm_free_cell_states(LSTMCellState *cs, int seq_len)` | Освобождение массива cell_states и всех буферов. |
| `void lstm_free_cell_state_buffers(LSTMCellState *cs, int seq_len)` | Освобождение только буферов (без самого массива). |

#### `include/model.h` / `src/model.c`

| Функция | Описание |
|---------|----------|
| `int model_save(...)` | Сохранение модели в текстовый формат. Секции: метаданные, `[normalization]`, `[lstm_weights]` (4 gate × recurrent/input/bias), `[output_weights]`. |
| `int model_load(const char *filepath, LSTMNetwork *net)` | Загрузка модели из текстового формата. Разбор по секциям и gate-ам. |

#### `include/types.h`

| Тип | Описание |
|-----|----------|
| `LSTMCellState` | Состояние ячейки на шаге t: hidden_state, cell_state, forget/input/output gate, candidate, concatenated. Сохраняется для BPTT. |

#### `main.c` — CLI

| Аргумент | Описание |
|----------|----------|
| `--cells N` | Количество LSTM ячеек (по умолчанию 128) |
| `--epochs N` | Количество эпох (по умолчанию 100) |
| `--lr F` | Learning rate (по умолчанию 0.0005) |
| `--seq N` | Длина последовательности (по умолчанию 64) |
| `--data PATH` | Путь к CSV-файлу |
| `--model PATH` | Путь к файлу модели |
| `--line N` | Начальная строка для predict/real режима |
| `--predict N` | Режим предсказания: N шагов вперёд |
| `--real N` | Режим реальных данных: N баров |

**Режимы работы:**
1. **Обучение** (по умолчанию): загружает данные, строит features, нормализует, делит 80/20, обучает, сохраняет модель каждую эпоху.
2. **Предсказание** (`--line N --predict M`): 64 референсных бара + M autoregressive прогнозных шагов.
3. **Реальные данные** (`--line N --real M`): вывод OHLC для M баров.

### Тесты

14 тестов, все проходят:

| # | Тест | Описание |
|---|------|----------|
| 1 | `test_forward_pass` | Forward pass без NaN/Inf |
| 2 | `test_concatenated_storage` | Bug #1: concatenated хранит все n+f элементов |
| 3 | `test_state_propagation` | H и C различаются между шагами |
| 4 | `test_backward_pass` | Градиенты ненулевые после BPTT |
| 5 | `test_last_timestep_grad` | Bug #2: градиенты на последнем шаге ненулевые |
| 6 | `test_output_grad_accumulation` | Bug #3: аккумуляция градиентов output layer |
| 7 | `test_update` | Веса изменяются после update |
| 8 | `test_batch_normalization` | Bug #6: batch_size нормализация |
| 9 | `test_training_loop` | MSE уменьшается на синтетических данных |
| 10 | `test_save_load` | Save/Load сохраняет предсказания |
| 11 | `test_deterministic_forward_64` | 64 cells × 64 steps × 6 features: H, C совпадают с аналитикой (err ~1e-16) |
| 12 | `test_bptt_gradient_check_64` | 64 cells × 5 steps: 8 gradient checks, max rel_err ~1e-6 |
| 13 | `test_deterministic_forward_128` | 128 cells × 256 steps × 12 features: H, C совпадают с аналитикой (err ~1e-14) |
| 14 | `test_bptt_gradient_check_128` | 128 cells × 4 steps × 12 features: 10 gradient checks, max rel_err ~1e-8 |

### Исправленные баги

| # | Баг | Исправление |
|---|-----|-------------|
| 1 | `concatenated` сохранял только n элементов вместо n+f | Отдельный цикл для всех combined элементов |
| 2 | `dcell` не аккумулировался на последнем шаге | Убран `if (t < seq_len - 1)` |
| 3 | Output layer: `=` вместо `+=` | `+=` для аккумуляции |
| 4 | `dcell += ... * forget_g` вместо `dcell *= forget_g` | Умножение для пропагации через cell state |
| 5 | `weight_*_input` матрицы не используются | Оставлено (не влияет на корректность) |
| 6 | Градиенты не нормализовались на batch_size | `inv_batch` в `lstm_update` |
| 7 | Дублирующееся объявление `lstm_update` | Удалён дубликат |
| 8 | C99-style объявления после кода | Перенесены в начало блока |
| 9 | **Нет sigmoid derivative для output gate** в BPTT | Добавлено `o*(1-o)` |


## 🇬🇧 English description

This project implements an LSTM (Long Short-Term Memory) neural network in C90 standard for forecasting financial time series (EUR/USD currency pair).

### Architecture

```
Input (6 features × 64 steps) → LSTM (128 cells) → LINEAR (128→1) → Prediction
```

- **Language standard:** C90 (no C99 features)
- **LSTM cells:** 128
- **Features:** 6
- **Time steps:** 64
- **Output:** regression (normalized close_return)
- **Loss function:** MSE (0.5 × (pred - target)²)

### Features (6)

| # | Name | Description |
|---|------|-------------|
| 0 | `candle_return` | (close - open) / open |
| 1 | `candle_range` | (high - low) / open |
| 2 | `close_return` | (close_t - close_{t-1}) / close_{t-1} |
| 3 | `volume_return` | (vol_t - vol_{t-1}) / vol_{t-1} |
| 4 | `ma_diff` | (MA5 - MA20) / MA20 |
| 5 | `volatility` | std(close_return, 12 bars) |

### Training parameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.0005 |
| Sequence length | 64 |
| Batch size | 32 |
| Epochs | 100 |
| Momentum (SGD) | 0.9 |
| Weight decay (L2) | 1e-6 |
| Gradient clipping | [-5.0, +5.0] |
| Xavier/Glorot init | sqrt(2/(fan_in + fan_out)) |

### Project structure

```
├── CMakeLists.txt           # CMake configuration (C90)
├── main.c                   # Entry point, CLI, training loop
├── include/
│   ├── types.h              # LSTMCellState — cell state for BPTT
│   ├── utils.h              # Matrices, math, random, clipping
│   ├── data.h               # DataRow, Dataset, loading/normalization
│   ├── lstm.h               # LSTMNetwork — network architecture
│   └── model.h              # Model save/load
├── src/
│   ├── utils.c              # Utility implementations
│   ├── data.c               # CSV parsing, feature engineering, split
│   ├── lstm.c               # Forward pass, BPTT, weight update
│   └── model.c              # Text format model I/O
├── tests/
│   └── test_lstm.c          # 14 tests (including gradient check)
├── resources/
│   ├── EURUSD_H1_*.csv      # Data (101053 H1 bars)
│   └── model.txt            # Trained model
└── scripts/
    ├── build.ps1            # Build via CMake + MinGW
    ├── train.ps1            # Training (Quick/Medium/Full)
    └── predict.ps1          # Prediction
```

### API — description of all functions

#### `include/utils.h` / `src/utils.c`

**Memory management:**

| Function | Description |
|----------|-------------|
| `void *utils_safe_malloc(size_t n)` | Allocate memory with check. On failure — `exit(EXIT_FAILURE)`. |
| `void *utils_safe_calloc(size_t n, size_t s)` | Allocate and zero memory with check. |
| `void utils_safe_free(void *ptr)` | Free memory (wrapper around `free`). |

**Matrix operations** (type `Matrix { double *data; int rows, cols; }`, row-major):

| Function | Description |
|----------|-------------|
| `Matrix *matrix_alloc(int rows, int cols)` | Create matrix filled with zeros. |
| `void matrix_free(Matrix *m)` | Free matrix. |
| `void matrix_set(Matrix *m, int r, int c, double val)` | Set element `m[r][c] = val`. |
| `double matrix_get(Matrix *m, int r, int c)` | Get element `m[r][c]`. |
| `void matrix_fill(Matrix *m, double val)` | Fill all elements with `val`. |
| `void matrix_copy(Matrix *dest, Matrix *src)` | Element-wise copy. |
| `void matrix_add(Matrix *dest, Matrix *src)` | Element-wise add `dest += src`. |
| `void matrix_sub(Matrix *dest, Matrix *src)` | Element-wise subtract `dest -= src`. |
| `void matrix_scale(Matrix *m, double val)` | Scale `m *= val`. |
| `void matrix_elem_mul(Matrix *dest, Matrix *a, Matrix *b)` | Element-wise multiply. |
| `void matrix_transpose(Matrix *dest, Matrix *src)` | Transpose. |
| `void matrix_mul(Matrix *dest, Matrix *a, Matrix *b)` | Matrix multiply `dest = a × b`. |
| `double matrix_dot_product(Matrix *a, Matrix *b)` | Dot product. |

**Math functions:**

| Function | Description |
|----------|-------------|
| `double utils_sigmoid(double x)` | Numerically stable sigmoid: `1/(1+e^{-x})` for x≥0, `e^x/(1+e^x)` for x<0. |
| `double utils_tanh_act(double x)` | Numerically stable tanh: clips \|x\|>20, else `(e^{2x}-1)/(e^{2x}+1)`. |
| `double utils_relu(double x)` | ReLU: `max(0, x)`. |
| `double utils_binary_cross_entropy(double prediction, double target)` | Binary cross-entropy loss with epsilon-clip. |

**Random number generation:**

| Function | Description |
|----------|-------------|
| `void utils_srand(unsigned int seed)` | Initialize RNG. |
| `double utils_rand(void)` | Uniform distribution [0, 1). |
| `double utils_gaussian_random(double mean, double stddev)` | Normal distribution (Box-Muller), clip \|z\|>10. |

**Other:**

| Function | Description |
|----------|-------------|
| `double utils_clip(double val, double min_val, double max_val)` | Clip value to [min, max]. |
| `void utils_trim(char *str)` | Remove leading/trailing whitespace. |
| `char *utils_strdup(const char *s)` | Duplicate string (C90-compatible). |

#### `include/data.h` / `src/data.c`

**Data types:**
- **`DataRow`** — one CSV record: date, time, OHLC, tickvol, vol, spread.
- **`Dataset`** — dataset: features, labels, normalization parameters.
- **`DatasetSplit`** — train/test split.

| Function | Description |
|----------|-------------|
| `int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count)` | Load CSV (MetaTrader format, tab-delimited). Skips header. Returns 0 on success. |
| `Dataset *data_build_dataset(DataRow *raw, int num_raw, int seq_len)` | Feature engineering: computes 6 features per bar, creates sliding windows of length `seq_len`, labels = next bar's close_return. Min-max normalization per feature. |
| `void data_normalize(Dataset *ds)` | Normalize features to [0, 1] using feat_min/feat_max. |
| `DatasetSplit data_split_train_test(Dataset *ds, double train_ratio)` | Chronological split (no shuffling). Train and test share feature/label pointers (shallow copy). |
| `void data_free_dataset(Dataset *ds)` | Free all Dataset resources. |
| `void data_free_split(DatasetSplit *split)` | Free split structures (without freeing shared data). |
| `int data_save_normalization(const char *filepath, Dataset *ds)` | Save normalization parameters to text file. |
| `int data_load_normalization(const char *filepath, Dataset *ds)` | Load normalization parameters. |

#### `include/lstm.h` / `src/lstm.c`

**Type `LSTMNetwork`** — for each gate (forget, input, output, candidate) contains:
- `weight_*_recurrent` — matrix [n × (n+f)] (recurrent + input weights)
- `weight_*_input` — matrix [n × f] (reserved, unused)
- `bias_*` — vector [n]
- `dweight_*`, `dbias_*` — gradients
- `mweight_*`, `mbias_*` — momentum

Plus output layer: `weight_output_layer` [1 × n], `bias_output_layer`.

| Function | Description |
|----------|-------------|
| `LSTMNetwork *lstm_init(int num_cells, int num_features, unsigned int seed)` | Create network with Xavier/Glorot initialization. scale = sqrt(2/(fan_in + num_cells)). All gradients and momentum zeroed. |
| `void lstm_free(LSTMNetwork *net)` | Free all network resources. |
| `double lstm_forward(LSTMNetwork *net, const double *sequence, LSTMCellState *cell_states, int seq_len)` | **Forward pass.** For each step t: computes forget/input/output/candidate gates, updates cell state `c_t = f_t·c_{t-1} + i_t·g_t`, hidden state `h_t = o_t·tanh(c_t)`. Stores all intermediate values in `cell_states[t]` for BPTT. Returns `bias_out + Σ W_out[i]·h[i]` (linear output). |
| `void lstm_backward(LSTMNetwork *net, const double *sequence, const LSTMCellState *cell_states, double prediction, double target, int seq_len)` | **Backpropagation through time (BPTT).** Computes gradients for all weights. Starts with `dL/dpred = pred - target`, propagates through output layer, then goes from t=seq_len-1 to t=0. For each step: gradients w.r.t. output/forget/input/candidate gates, accumulates weight gradients, propagates `dhidden` and `dcell` to previous step (`dcell = dcell * forget_gate`). |
| `void lstm_zero_gradients(LSTMNetwork *net)` | Zero all gradients (call before each batch). |
| `void lstm_update(LSTMNetwork *net, double lr, double momentum, int batch_size)` | **SGD + momentum.** Gradients normalized by batch_size, clipped to [-5, +5], then: `v = momentum·v - lr·(grad + weight_decay·weight)`, `weight += v`. |
| `double lstm_compute_loss(LSTMNetwork *net, Dataset *ds)` | MSE loss over entire dataset: `Σ 0.5·(pred - label)² / N`. |
| `double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds)` | Direction accuracy: fraction where `sign(pred) == sign(label)`. |
| `double lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len)` | Inference: allocates cell_states, calls `lstm_forward`, frees. |
| `void lstm_free_cell_states(LSTMCellState *cs, int seq_len)` | Free cell_states array and all buffers. |
| `void lstm_free_cell_state_buffers(LSTMCellState *cs, int seq_len)` | Free only buffers (not the array itself). |

#### `include/model.h` / `src/model.c`

| Function | Description |
|----------|-------------|
| `int model_save(...)` | Save model to text format. Sections: metadata, `[normalization]`, `[lstm_weights]` (4 gates × recurrent/input/bias), `[output_weights]`. |
| `int model_load(const char *filepath, LSTMNetwork *net)` | Load model from text format. Parses by sections and gates. |

#### `include/types.h`

| Type | Description |
|------|-------------|
| `LSTMCellState` | Cell state at step t: hidden_state, cell_state, forget/input/output gate, candidate, concatenated. Stored for BPTT. |

#### `main.c` — CLI

| Argument | Description |
|----------|-------------|
| `--cells N` | Number of LSTM cells (default 128) |
| `--epochs N` | Number of epochs (default 100) |
| `--lr F` | Learning rate (default 0.0005) |
| `--seq N` | Sequence length (default 64) |
| `--data PATH` | Path to CSV file |
| `--model PATH` | Path to model file |
| `--line N` | Starting row for predict/real mode |
| `--predict N` | Prediction mode: N steps ahead |
| `--real N` | Real data mode: N bars |

**Modes:**
1. **Training** (default): loads data, builds features, normalizes, splits 80/20, trains, saves model every epoch.
2. **Prediction** (`--line N --predict M`): 64 reference bars + M autoregressive prediction steps.
3. **Real data** (`--line N --real M`): output OHLC for M bars starting at row N.

### Tests

14 tests, all passing:

| # | Test | Description |
|---|------|-------------|
| 1 | `test_forward_pass` | Forward pass produces no NaN/Inf |
| 2 | `test_concatenated_storage` | Bug #1: concatenated stores all n+f elements |
| 3 | `test_state_propagation` | H and C differ between timesteps |
| 4 | `test_backward_pass` | Gradients are non-zero after BPTT |
| 5 | `test_last_timestep_grad` | Bug #2: gradients at last timestep are non-zero |
| 6 | `test_output_grad_accumulation` | Bug #3: output layer gradient accumulation |
| 7 | `test_update` | Weights change after update |
| 8 | `test_batch_normalization` | Bug #6: batch_size normalization |
| 9 | `test_training_loop` | MSE decreases on synthetic data |
| 10 | `test_save_load` | Save/Load preserves predictions |
| 11 | `test_deterministic_forward_64` | 64 cells × 64 steps × 6 features: H, C match analytical solution (err ~1e-16) |
| 12 | `test_bptt_gradient_check_64` | 64 cells × 5 steps: 8 gradient checks, max rel_err ~1e-6 |
| 13 | `test_deterministic_forward_128` | 128 cells × 256 steps × 12 features: H, C match analytical solution (err ~1e-14) |
| 14 | `test_bptt_gradient_check_128` | 128 cells × 4 steps × 12 features: 10 gradient checks, max rel_err ~1e-8 |

### Fixed bugs

| # | Bug | Fix |
|---|-----|-----|
| 1 | `concatenated` stored only n elements instead of n+f | Separate loop for all combined elements |
| 2 | `dcell` not accumulated at last timestep | Removed `if (t < seq_len - 1)` |
| 3 | Output layer: `=` instead of `+=` | `+=` for accumulation |
| 4 | `dcell += ... * forget_g` instead of `dcell *= forget_g` | Multiplication for cell state propagation |
| 5 | `weight_*_input` matrices unused | Left as-is (no correctness impact) |
| 6 | Gradients not normalized by batch_size | `inv_batch` in `lstm_update` |
| 7 | Duplicate `lstm_update` declaration | Removed duplicate |
| 8 | C99-style declarations after code | Moved to block start |
| 9 | **Missing sigmoid derivative for output gate** in BPTT | Added `o*(1-o)` |
