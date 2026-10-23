#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Constants ==================== */
#define INITIAL_CAPACITY 1024
#define CAPACITY_MULTIPLIER 2
#define DATE_TIME_LEN 16
#define LINE_BUFFER_SIZE 4096
#define TRAIN_RATIO 0.8
#define NORMALIZATION_EPSILON 1e-15

/* ==================== CSV Parser ==================== */

static char *skip_quotes(char *p) {
    while (*p == '"' || *p == ' ' || *p == '\t') p++;
    return p;
}

static char *find_field_end(char *p) {
    if (*p == '"') {
        p++;
        while (*p && *p != '"') p++;
        return (*p == '"') ? p + 1 : p;
    }
    while (*p && *p != '\t' && *p != '\n' && *p != '\r') p++;
    return p;
}

int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count) {
    FILE *f;
    int capacity;
    int count;
    DataRow *rows;
    char line[4096];
    char *p;
    char *end;
    int len;

    f = fopen(filepath, "r");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filepath);
        return -1;
    }

    capacity = INITIAL_CAPACITY;
    count = 0;
    rows = (DataRow *)malloc((size_t)capacity * sizeof(DataRow));
    if (!rows) {
        fclose(f);
        return -1;
    }

    {
        int header_skipped = 0;

        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;

            if (!header_skipped) {
                header_skipped = 1;
                continue;
            }

            if (count >= capacity) {
                capacity *= CAPACITY_MULTIPLIER;
                rows = (DataRow *)realloc(rows, (size_t)capacity * sizeof(DataRow));
                if (!rows) {
                    fclose(f);
                    return -1;
                }
            }

            p = line;

            /* <DATE> */
            p = skip_quotes(p);
            end = find_field_end(p);
            len = (int)(end - p);
            if (len >= 16) len = 15;
            memcpy(rows[count].date, p, (size_t)len);
            rows[count].date[len] = 0;
            p = end;

            /* <TIME> */
            p = skip_quotes(p);
            end = find_field_end(p);
            len = (int)(end - p);
            if (len >= 16) len = 15;
            memcpy(rows[count].time_, p, (size_t)len);
            rows[count].time_[len] = 0;
            p = end;

            /* <OPEN> */
            p = skip_quotes(p);
            rows[count].open = atof(p);

            /* <HIGH> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].high = atof(p);

            /* <LOW> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].low = atof(p);

            /* <CLOSE> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].close = atof(p);

            /* <TICKVOL> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].tickvol = atoi(p);

            /* <VOL> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].vol = atoi(p);

            /* <SPREAD> */
            p = find_field_end(p);
            p = skip_quotes(p);
            rows[count].spread = atoi(p);

            count++;
        }
    }

    fclose(f);
    *out_rows = rows;
    *out_count = count;
    return 0;
}

/* ==================== Feature Engineering ==================== */

Dataset *data_build_dataset(DataRow *raw, int num_raw, int seq_len) {
    int i, j, s, fi;
    int num_features = 6;
    const char feat_names[][32] = {
        "candle_return", "candle_range", "close_return",
        "volume_return", "ma_diff", "volatility"
    };
    int num_timesteps;
    int num_samples;
    Dataset *ds;
    double **raw_features;
    double cr, rng, cret, vol_ret, ma_diff, vol;
    double sum, mean, std_dev;
    int window;
    int t_start, t_end, label_idx;
    int ti;

    num_timesteps = num_raw - 1;
    num_samples = num_timesteps - seq_len;
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Not enough data. Need at least %d rows, have %d\n",
                seq_len + 2, num_raw);
        return NULL;
    }

    ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_samples = num_samples;
    ds->num_features = num_features;
    ds->sequence_length = seq_len;
    ds->raw_data = raw;
    ds->num_raw = num_raw;

    ds->features = (double **)malloc((size_t)num_samples * sizeof(double *));
    ds->labels = (double *)malloc((size_t)num_samples * sizeof(double));
    ds->feat_min = (double *)malloc((size_t)num_features * sizeof(double));
    ds->feat_max = (double *)malloc((size_t)num_features * sizeof(double));
    ds->feat_names = (char **)malloc((size_t)num_features * sizeof(char *));

    for (j = 0; j < num_features; j++) {
        ds->feat_names[j] = utils_strdup(feat_names[j]);
        ds->feat_min[j] = 1e30;
        ds->feat_max[j] = -1e30;
    }

    raw_features = (double **)malloc((size_t)num_timesteps * sizeof(double *));
    for (i = 0; i < num_timesteps; i++) {
        raw_features[i] = (double *)malloc((size_t)num_features * sizeof(double));
    }

    for (i = 0; i < num_timesteps; i++) {
        /* Feature 0: candle_return */
        cr = (raw[i].close - raw[i].open) / raw[i].open;
        
        /* Feature 1: candle_range */
        rng = (raw[i].high - raw[i].low) / raw[i].open;
        
        /* Feature 2: close_return */
        cret = (i > 0) ? (raw[i].close - raw[i - 1].close) / raw[i - 1].close : 0.0;
        
        /* Feature 3: volume_return (change in tickvol) */
        vol_ret = (i > 0 && raw[i - 1].tickvol > 0) ?
            (raw[i].tickvol - raw[i - 1].tickvol) / (double)raw[i - 1].tickvol :
            0.0;
        
        /* Feature 4: ma_diff (MA5 - MA20) */
        ma_diff = 0.0;
        if (i >= 20) {
            double sum_short = 0.0, sum_long = 0.0;
            int k;
            for (k = 0; k < 5; k++) {
                sum_short += raw[i - k].close;
            }
            for (k = 0; k < 20; k++) {
                sum_long += raw[i - k].close;
            }
            sum_short /= 5.0;
            sum_long /= 20.0;
            ma_diff = (sum_short - sum_long) / sum_long;
        }
        
        /* Feature 5: volatility (std dev of close_return over last 12 candles) */
        vol = 0.0;
        if (i >= 12) {
            double sum_ret = 0.0;
            int ret_count;
            int m;
            
            /* Calculate mean return */
            ret_count = 0;
            for (m = i - 12; m < i; m++) {
                if (m > 0) {
                    sum_ret += (raw[m].close - raw[m - 1].close) / raw[m - 1].close;
                    ret_count++;
                }
            }
            
            if (ret_count > 0) {
                mean = sum_ret / (double)ret_count;
                sum = 0.0;
                for (m = i - 12; m < i; m++) {
                    if (m > 0) {
                        double r = (raw[m].close - raw[m - 1].close) / raw[m - 1].close;
                        sum += (r - mean) * (r - mean);
                    }
                }
                std_dev = sqrt(sum / (double)ret_count);
                vol = std_dev;
            }
        }
        
        raw_features[i][0] = cr;
        raw_features[i][1] = rng;
        raw_features[i][2] = cret;
        raw_features[i][3] = vol_ret;
        raw_features[i][4] = ma_diff;
        raw_features[i][5] = vol;

        for (j = 0; j < num_features; j++) {
            if (raw_features[i][j] < ds->feat_min[j])
                ds->feat_min[j] = raw_features[i][j];
            if (raw_features[i][j] > ds->feat_max[j])
                ds->feat_max[j] = raw_features[i][j];
        }
    }

    for (i = 0; i < num_samples; i++) {
        t_start = i;
        t_end = i + seq_len - 1;
        label_idx = t_end + 1;

        ds->features[i] = (double *)malloc((size_t)(seq_len * num_features) * sizeof(double));
        /* Regression label: normalized close_return for next candle */
        ds->labels[i] = (raw[label_idx].close - raw[t_end].close) / raw[t_end].close;

        for (s = 0; s < seq_len; s++) {
            ti = t_start + s;
            for (fi = 0; fi < num_features; fi++) {
                ds->features[i][s * num_features + fi] = raw_features[ti][fi];
            }
        }
    }

    for (i = 0; i < num_timesteps; i++) {
        free(raw_features[i]);
    }
    free(raw_features);

    return ds;
}

void data_free_dataset(Dataset *ds) {
    int i, j;

    if (!ds) return;
    if (ds->features) {
        for (i = 0; i < ds->num_samples; i++) {
            free(ds->features[i]);
        }
        free(ds->features);
    }
    free(ds->labels);
    if (ds->feat_min) free(ds->feat_min);
    if (ds->feat_max) free(ds->feat_max);
    if (ds->feat_names) {
        for (j = 0; j < ds->num_features; j++) {
            free(ds->feat_names[j]);
        }
        free(ds->feat_names);
    }
    free(ds);
}

/* ==================== Normalization ==================== */

void data_normalize(Dataset *ds) {
    int i, j, s, fi;
    double range;

    for (i = 0; i < ds->num_samples; i++) {
        for (j = 0; j < ds->num_features; j++) {
            for (s = 0; s < ds->sequence_length; s++) {
                fi = s * ds->num_features + j;
                range = ds->feat_max[j] - ds->feat_min[j];
                if (range < NORMALIZATION_EPSILON) range = 1.0;
                ds->features[i][fi] =
                    (ds->features[i][fi] - ds->feat_min[j]) / range;
            }
        }
    }
}

/* ==================== Train/Test Split ==================== */

DatasetSplit data_split_train_test(Dataset *ds, double train_ratio) {
    DatasetSplit split;
    int train_count;
    int test_count;
    int i, src;

    split.train = NULL;
    split.test = NULL;

    train_count = (int)((double)ds->num_samples * TRAIN_RATIO);
    if (train_count < 1) train_count = 1;
    if (train_count >= ds->num_samples) train_count = ds->num_samples - 1;

    test_count = ds->num_samples - train_count;

    split.train = (Dataset *)malloc(sizeof(Dataset));
    split.test = (Dataset *)malloc(sizeof(Dataset));
    if (!split.train || !split.test) {
        free(split.train);
        free(split.test);
        split.train = NULL;
        split.test = NULL;
        return split;
    }

    *split.train = *ds;
    *split.test = *ds;

    split.train->num_samples = train_count;
    split.test->num_samples = test_count;

    split.train->features = (double **)malloc((size_t)train_count * sizeof(double *));
    split.train->labels = (double *)malloc((size_t)train_count * sizeof(double));
    split.test->features = (double **)malloc((size_t)test_count * sizeof(double *));
    split.test->labels = (double *)malloc((size_t)test_count * sizeof(double));

    if (!split.train->features || !split.train->labels ||
        !split.test->features || !split.test->labels) {
        free(split.train->features);
        free(split.train->labels);
        free(split.test->features);
        free(split.test->labels);
        free(split.train);
        free(split.test);
        split.train = NULL;
        split.test = NULL;
        return split;
    }

    for (i = 0; i < train_count; i++) {
        split.train->features[i] = ds->features[i];
        split.train->labels[i] = ds->labels[i];
    }
    for (i = 0; i < test_count; i++) {
        src = train_count + i;
        split.test->features[i] = ds->features[src];
        split.test->labels[i] = ds->labels[src];
    }

    split.train->feat_min = ds->feat_min;
    split.train->feat_max = ds->feat_max;
    split.train->feat_names = ds->feat_names;
    split.test->feat_min = ds->feat_min;
    split.test->feat_max = ds->feat_max;
    split.test->feat_names = ds->feat_names;

    split.train->raw_data = ds->raw_data;
    split.test->raw_data = ds->raw_data;
    split.train->num_raw = ds->num_raw;
    split.test->num_raw = ds->num_raw;

    return split;
}

/* ==================== Normalization I/O ==================== */

int data_save_normalization(const char *filepath, Dataset *ds) {
    FILE *f;
    int j;

    f = fopen(filepath, "w");
    if (!f) return -1;

    fprintf(f, "# Normalization Parameters\n");
    fprintf(f, "# num_features: %d\n", ds->num_features);
    fprintf(f, "# sequence_length: %d\n\n", ds->sequence_length);

    for (j = 0; j < ds->num_features; j++) {
        fprintf(f, "feature_name: %s | min: %.8f | max: %.8f\n",
                ds->feat_names[j], ds->feat_min[j], ds->feat_max[j]);
    }

    fclose(f);
    return 0;
}

int data_load_normalization(const char *filepath, Dataset *ds) {
    FILE *f;
    char line[512];
    int count = 0;
    char name[64];
    double mn, mx;

    f = fopen(filepath, "r");
    if (!f) return -1;

    while (fgets(line, sizeof(line), f) && count < ds->num_features) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

        if (sscanf(line, "feature_name: %63s | min: %lf | max: %lf", name, &mn, &mx) == 3) {
            if (count < ds->num_features) {
                free(ds->feat_names[count]);
                ds->feat_names[count] = utils_strdup(name);
                ds->feat_min[count] = mn;
                ds->feat_max[count] = mx;
                count++;
            }
        }
    }

    fclose(f);
    return (count == ds->num_features) ? 0 : -1;
}

/* ==================== Free Split ==================== */

void data_free_split(DatasetSplit *split) {
    int i;

    if (!split) return;
    if (split->train) {
        /* features and labels are shallow copies — do NOT free them */
        free(split->train);
    }
    if (split->test) {
        /* features and labels are shallow copies — do NOT free them */
        free(split->test);
    }
}
