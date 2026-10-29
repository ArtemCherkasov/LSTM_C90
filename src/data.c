#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Constants ==================== */
#define INITIAL_CAPACITY 1024
#define CAPACITY_MULTIPLIER 2
#define TRAIN_RATIO 0.8
#define NORMALIZATION_EPSILON 1e-15
#define VOLUME_RETURN_CLIP 10.0

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
    int capacity, count;
    DataRow *rows;
    char line[4096];
    char *p, *end;
    int len;

    f = fopen(filepath, "r");
    if (!f) { fprintf(stderr, "Error: Cannot open %s\n", filepath); return -1; }

    capacity = INITIAL_CAPACITY;
    count = 0;
    rows = (DataRow *)malloc((size_t)capacity * sizeof(DataRow));
    if (!rows) { fclose(f); return -1; }

    {
        int header_skipped = 0;
        while (fgets(line, sizeof(line), f)) {
            if (line[0] == '\n' || line[0] == '\r' || line[0] == 0) continue;
            if (!header_skipped) { header_skipped = 1; continue; }

            if (count >= capacity) {
                capacity *= CAPACITY_MULTIPLIER;
                rows = (DataRow *)realloc(rows, (size_t)capacity * sizeof(DataRow));
                if (!rows) { fclose(f); return -1; }
            }

            p = line;
            p = skip_quotes(p); end = find_field_end(p); len = (int)(end - p);
            if (len >= 16) len = 15; memcpy(rows[count].date, p, (size_t)len); rows[count].date[len] = 0; p = end;
            p = skip_quotes(p); end = find_field_end(p); len = (int)(end - p);
            if (len >= 16) len = 15; memcpy(rows[count].time_, p, (size_t)len); rows[count].time_[len] = 0; p = end;
            p = skip_quotes(p); rows[count].open = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].high = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].low = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].close = atof(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].tickvol = atoi(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].vol = atoi(p);
            p = find_field_end(p); p = skip_quotes(p); rows[count].spread = atoi(p);
            count++;
        }
    }
    fclose(f);
    *out_rows = rows;
    *out_count = count;
    return 0;
}

/* ==================== Feature Engineering (common) ==================== */

double **data_compute_raw_features(DataRow *raw, int num_raw, int num_features,
                                    double *feat_min, double *feat_max,
                                    char **feat_names) {
    int i, j;
    double **raw_features;
    double cr, rng, cret, vol_ret, ma_diff, vol;
    double sum, mean, std_dev;
    const char names[][32] = {
        "candle_return", "candle_range", "close_return",
        "volume_return", "ma_diff", "volatility"
    };

    raw_features = (double **)malloc((size_t)num_raw * sizeof(double *));
    for (i = 0; i < num_raw; i++)
        raw_features[i] = (double *)malloc((size_t)num_features * sizeof(double));

    for (j = 0; j < num_features; j++) {
        feat_names[j] = utils_strdup(names[j]);
        feat_min[j] = 1e30;
        feat_max[j] = -1e30;
    }

    for (i = 0; i < num_raw; i++) {
        cr = (raw[i].close - raw[i].open) / raw[i].open;
        rng = (raw[i].high - raw[i].low) / raw[i].open;
        cret = (i > 0) ? (raw[i].close - raw[i - 1].close) / raw[i - 1].close : 0.0;
        vol_ret = (i > 0 && raw[i - 1].tickvol > 0) ?
            (raw[i].tickvol - raw[i - 1].tickvol) / (double)raw[i - 1].tickvol : 0.0;
        if (vol_ret > VOLUME_RETURN_CLIP) vol_ret = VOLUME_RETURN_CLIP;
        if (vol_ret < -VOLUME_RETURN_CLIP) vol_ret = -VOLUME_RETURN_CLIP;
        ma_diff = 0.0;
        if (i >= 20) {
            double ss = 0.0, sl = 0.0; int kk;
            for (kk = 0; kk < 5; kk++) ss += raw[i - kk].close;
            for (kk = 0; kk < 20; kk++) sl += raw[i - kk].close;
            ma_diff = (ss / 5.0 - sl / 20.0) / (sl / 20.0);
        }
        vol = 0.0;
        if (i >= 12) {
            double sr = 0.0; int rc = 0, m;
            for (m = i - 12; m < i; m++) {
                if (m > 0) { sr += (raw[m].close - raw[m-1].close) / raw[m-1].close; rc++; }
            }
            if (rc > 0) {
                mean = sr / (double)rc; sum = 0.0;
                for (m = i - 12; m < i; m++) {
                    if (m > 0) { double r = (raw[m].close - raw[m-1].close) / raw[m-1].close; sum += (r - mean) * (r - mean); }
                }
                vol = sqrt(sum / (double)rc);
            }
        }
        raw_features[i][0] = cr; raw_features[i][1] = rng; raw_features[i][2] = cret;
        raw_features[i][3] = vol_ret; raw_features[i][4] = ma_diff; raw_features[i][5] = vol;
        for (j = 0; j < num_features; j++) {
            if (raw_features[i][j] < feat_min[j]) feat_min[j] = raw_features[i][j];
            if (raw_features[i][j] > feat_max[j]) feat_max[j] = raw_features[i][j];
        }
    }
    return raw_features;
}

void data_free_raw_features(double **raw_features, int num_raw) {
    int i;
    for (i = 0; i < num_raw; i++) free(raw_features[i]);
    free(raw_features);
}

/* ==================== Dataset Allocation (common) ==================== */

Dataset *data_alloc_dataset(DataRow *raw, int num_raw, int seq_len,
                             int prediction_horizon, int num_outputs, int mode) {
    int num_features = 6;
    int num_samples;
    Dataset *ds;
    double **raw_features;
    int i, s, fi, ti;

    num_samples = num_raw - seq_len - prediction_horizon;
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Not enough data. Need at least %d rows, have %d\n",
                seq_len + prediction_horizon + 1, num_raw);
        return NULL;
    }

    ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) return NULL;

    ds->num_samples = num_samples;
    ds->num_features = num_features;
    ds->num_outputs = num_outputs;
    ds->sequence_length = seq_len;
    ds->mode = mode;
    ds->raw_data = raw;
    ds->num_raw = num_raw;

    ds->features = (double **)malloc((size_t)num_samples * sizeof(double *));
    ds->labels = (double **)malloc((size_t)num_samples * sizeof(double *));
    ds->feat_min = (double *)malloc((size_t)num_features * sizeof(double));
    ds->feat_max = (double *)malloc((size_t)num_features * sizeof(double));
    ds->feat_names = (char **)malloc((size_t)num_features * sizeof(char *));
    ds->label_min = (double *)malloc((size_t)num_outputs * sizeof(double));
    ds->label_max = (double *)malloc((size_t)num_outputs * sizeof(double));

    raw_features = data_compute_raw_features(raw, num_raw, num_features,
                                              ds->feat_min, ds->feat_max, ds->feat_names);

    for (i = 0; i < num_outputs; i++) {
        ds->label_min[i] = 1e30;
        ds->label_max[i] = -1e30;
    }

    for (i = 0; i < num_samples; i++) {
        ds->features[i] = (double *)malloc((size_t)(seq_len * num_features) * sizeof(double));
        ds->labels[i] = (double *)malloc((size_t)(seq_len * num_outputs) * sizeof(double));
        for (s = 0; s < seq_len; s++) {
            ti = i + s;
            for (fi = 0; fi < num_features; fi++)
                ds->features[i][s * num_features + fi] = raw_features[ti][fi];
        }
    }

    data_free_raw_features(raw_features, num_raw);
    return ds;
}

/* ==================== Free Dataset ==================== */

void data_free_dataset(Dataset *ds) {
    int i, j;
    if (!ds) return;
    if (ds->features) { for (i = 0; i < ds->num_samples; i++) free(ds->features[i]); free(ds->features); }
    if (ds->labels) { for (i = 0; i < ds->num_samples; i++) free(ds->labels[i]); free(ds->labels); }
    if (ds->feat_min) free(ds->feat_min);
    if (ds->feat_max) free(ds->feat_max);
    if (ds->feat_names) { for (j = 0; j < ds->num_features; j++) free(ds->feat_names[j]); free(ds->feat_names); }
    if (ds->label_min) free(ds->label_min);
    if (ds->label_max) free(ds->label_max);
    free(ds);
}

/* ==================== Feature Normalization (common) ==================== */

void data_normalize_features(Dataset *ds) {
    int i, j, s, fi;
    double range;
    for (i = 0; i < ds->num_samples; i++) {
        for (j = 0; j < ds->num_features; j++) {
            for (s = 0; s < ds->sequence_length; s++) {
                fi = s * ds->num_features + j;
                range = ds->feat_max[j] - ds->feat_min[j];
                if (range < NORMALIZATION_EPSILON) range = 1.0;
                ds->features[i][fi] = (ds->features[i][fi] - ds->feat_min[j]) / range;
            }
        }
    }
}

/* ==================== Train/Test Split ==================== */

DatasetSplit data_split_train_test(Dataset *ds, double train_ratio) {
    DatasetSplit split;
    int train_count, test_count, i, src;
    split.train = NULL; split.test = NULL;
    train_count = (int)((double)ds->num_samples * TRAIN_RATIO);
    if (train_count < 1) train_count = 1;
    if (train_count >= ds->num_samples) train_count = ds->num_samples - 1;
    test_count = ds->num_samples - train_count;

    split.train = (Dataset *)malloc(sizeof(Dataset));
    split.test = (Dataset *)malloc(sizeof(Dataset));
    if (!split.train || !split.test) { free(split.train); free(split.test); split.train = NULL; split.test = NULL; return split; }
    *split.train = *ds; *split.test = *ds;
    split.train->num_samples = train_count; split.test->num_samples = test_count;

    split.train->features = (double **)malloc((size_t)train_count * sizeof(double *));
    split.train->labels = (double **)malloc((size_t)train_count * sizeof(double *));
    split.test->features = (double **)malloc((size_t)test_count * sizeof(double *));
    split.test->labels = (double **)malloc((size_t)test_count * sizeof(double *));
    if (!split.train->features || !split.train->labels || !split.test->features || !split.test->labels) {
        free(split.train->features); free(split.train->labels); free(split.test->features); free(split.test->labels);
        free(split.train); free(split.test); split.train = NULL; split.test = NULL; return split;
    }
    for (i = 0; i < train_count; i++) { split.train->features[i] = ds->features[i]; split.train->labels[i] = ds->labels[i]; }
    for (i = 0; i < test_count; i++) { src = train_count + i; split.test->features[i] = ds->features[src]; split.test->labels[i] = ds->labels[src]; }

    split.train->feat_min = ds->feat_min; split.train->feat_max = ds->feat_max; split.train->feat_names = ds->feat_names;
    split.train->label_min = ds->label_min; split.train->label_max = ds->label_max;
    split.test->feat_min = ds->feat_min; split.test->feat_max = ds->feat_max; split.test->feat_names = ds->feat_names;
    split.test->label_min = ds->label_min; split.test->label_max = ds->label_max;
    split.train->raw_data = ds->raw_data; split.test->raw_data = ds->raw_data;
    split.train->num_raw = ds->num_raw; split.test->num_raw = ds->num_raw;
    return split;
}

/* ==================== Normalization I/O ==================== */

int data_save_normalization(const char *filepath, Dataset *ds) {
    FILE *f; int j;
    f = fopen(filepath, "w"); if (!f) return -1;
    fprintf(f, "# Normalization Parameters\n# num_features: %d\n# num_outputs: %d\n# sequence_length: %d\n\n", ds->num_features, ds->num_outputs, ds->sequence_length);
    for (j = 0; j < ds->num_features; j++)
        fprintf(f, "feature_name: %s | min: %.8f | max: %.8f\n", ds->feat_names[j], ds->feat_min[j], ds->feat_max[j]);
    fprintf(f, "label_min:"); for (j = 0; j < ds->num_outputs; j++) fprintf(f, " %.8f", ds->label_min[j]); fprintf(f, "\n");
    fprintf(f, "label_max:"); for (j = 0; j < ds->num_outputs; j++) fprintf(f, " %.8f", ds->label_max[j]); fprintf(f, "\n");
    fclose(f); return 0;
}

int data_load_normalization(const char *filepath, Dataset *ds) {
    FILE *f; char line[8192]; int count = 0; char name[64]; double mn, mx;
    f = fopen(filepath, "r"); if (!f) return -1;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (strncmp(line, "label_min:", 10) == 0) { char *p = line + 10; int j; for (j = 0; j < ds->num_outputs && *p; j++) ds->label_min[j] = strtod(p, &p); continue; }
        if (strncmp(line, "label_max:", 10) == 0) { char *p = line + 10; int j; for (j = 0; j < ds->num_outputs && *p; j++) ds->label_max[j] = strtod(p, &p); continue; }
        if (sscanf(line, "feature_name: %63s | min: %lf | max: %lf", name, &mn, &mx) == 3) {
            if (count < ds->num_features) { free(ds->feat_names[count]); ds->feat_names[count] = utils_strdup(name); ds->feat_min[count] = mn; ds->feat_max[count] = mx; count++; }
        }
    }
    fclose(f); return (count == ds->num_features) ? 0 : -1;
}

/* ==================== Free Split ==================== */

void data_free_split(DatasetSplit *split) {
    if (!split) return;
    if (split->train) free(split->train);
    if (split->test) free(split->test);
}
