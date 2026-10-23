#ifndef DATA_H
#define DATA_H

#include "utils.h"

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
    double *labels;      /* labels[sample_idx] */
    int num_samples;
    int num_features;
    int sequence_length;

    /* Normalization parameters */
    double *feat_min;
    double *feat_max;
    char **feat_names;

    /* Raw data for reference */
    DataRow *raw_data;
    int num_raw;
} Dataset;

/* ==================== Functions ==================== */

/* Load raw data from MetaTrader CSV (tab-delimited, quoted) */
int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count);

/* Build features, labels, normalization params, and sequences */
Dataset *data_build_dataset(DataRow *raw, int num_raw, int seq_len);

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

/* Normalize dataset in-place using feat_min/feat_max */
void data_normalize(Dataset *ds);

/* Free a train/test split (does not free shared normalization params) */
void data_free_split(DatasetSplit *split);

#endif /* DATA_H */
