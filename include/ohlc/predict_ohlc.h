#ifndef PREDICT_OHLC_H
#define PREDICT_OHLC_H

#include "lstm.h"
#include "data.h"

/* Print OHLC predictions%te outputs from last timestep */
void predict_ohlc(double *pred_outputs, int num_outputs, int seq_len,
                  double *lbl_min_arr, double *lbl_max_arr,
                  double ref_close, int ohlc_predict_count);

/* Demo: predict on last few samples, print OHLC */
void demo_ohlc(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count);

#endif /* PREDICT_OHLC_H */
