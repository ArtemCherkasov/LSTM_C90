#ifndef PREDICT_OHLC_H
#define PREDICT_OHLC_H

#include "lstm.h"
#include "data.h"

/* Print OHLC predictions: takes outputs from last timestep */
/* raw_data and end_row provide date-time for each predicted candle */
void predict_ohlc(double *pred_outputs, int num_outputs, int seq_len,
                  double *lbl_min_arr, double *lbl_max_arr,
                  double ref_close, int ohlc_predict_count,
                  DataRow *raw_data, int end_row);

/* Demo: predict on last few samples, print OHLC */
void demo_ohlc(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count);

#endif /* PREDICT_OHLC_H */
