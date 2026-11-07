#ifndef PREDICT_DIRECTION_H
#define PREDICT_DIRECTION_H

#include "lstm.h"
#include "data.h"

/* Print direction predictions: threshold at 0.5 -> 1 (up) or 0 (down) */
/* raw_data and end_row provide date-time for each predicted candle */
void predict_direction(double *pred_outputs, int num_outputs, int seq_len,
                        int ohlc_predict_count,
                        DataRow *raw_data, int end_row);

/* Demo: predict on last few samples, print directions */
void demo_direction(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count);

#endif /* PREDICT_DIRECTION_H */
