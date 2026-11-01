#ifndef DATA_H
#define DATA_H

#include "utils.h"

/* ==================== Mode constants ==================== */
#define MODE_OHLC 0
#define MODE_DIRECTION 1

/* ==================== Feature count ==================== */
#define NUM_RAW_FEATURES 26

/* ==================== Raw Data Row ==================== */
typedef struct {
    char date[16];
    char time_[16];
    double open;
    double high;
    double low;
    double close;
    int tickvol;
    int vol;
    int spread;
} DataRow;

/* ==================== Dataset ==================== */
typedef struct {
    double **features;   /* features[sample_idx][feature_dim] */
    double **labels;     /* labels[sample_idx][sequence_length * num_outputs] */
    int num_samples;
    int num_features;
    int num_outputs;
    int sequence_length;
    int mode;            /* 0=ohlc, 1=direction */

    /* Normalization parameters */
    double *feat_min;
    double *feat_max;
    char **feat_names;

    /* Label normalization (symmetric scaling by max_abs, per output) */
    double *label_min;
    double *label_max;

    /* Raw data for reference */
    DataRow *raw_data;
    int num_raw;
} Dataset;

/* ==================== Common Functions ==================== */

/* Load raw data from MetaTrader CSV (tab-delimited, quoted) */
int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count);

/* Aggregate raw H1 candles into higher timeframe (h_period candles -> 1) */
/* Open=first, High=max, Low=min, Close=last, TickVol=sum */
void data_aggregate_raw(DataRow *raw, int num_raw, int h_period,
                        DataRow **out_raw, int *out_count);

/* Compute NUM_RAW_FEATURES features for all rows. */
/* Also fills feat_min/feat_max and feat_names in the dataset. */
double **data_compute_raw_features(DataRow *raw, int num_raw, int num_features,
                                    double *feat_min, double *feat_max,
                                    char **feat_names);

/* Free raw features array */
void data_free_raw_features(double **raw_features, int num_raw);

/* Allocate Dataset struct with features filled (no labels yet) */
Dataset *data_alloc_dataset(DataRow *raw, int num_raw, int seq_len,
                             int prediction_horizon, int num_outputs, int mode);

/* Free dataset */
void data_free_dataset(Dataset *ds);

/* Create training/test split (80/20 chronological) */
typedef struct {
    Dataset *train;
    Dataset *test;
} DatasetSplit;

DatasetSplit data_split_train_test(Dataset *ds, double train_ratio);

/* Save normalization params to text file */
int data_save_normalization(const char *filepath, Dataset *ds);

/* Load normalization params from text file */
int data_load_normalization(const char *filepath, Dataset *ds);

/* Normalize features in-place using feat_min/feat_max (common) */
void data_normalize_features(Dataset *ds);

/* Free a train/test split (does not free shared normalization params) */
void data_free_split(DatasetSplit *split);

#endif /* DATA_H */
