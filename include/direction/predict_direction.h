#ifndef PREDICT_DIRECTION_H
#define PREDICT_DIRECTION_H

#include "lstm.h"
#include "data.h"

/* Print direction predictions: threshold at 0 -> 1 or -1 */
void predict_direction(double *pred_outputs, int num_outputs, int seq_len,
                        int ohlc_predict_count);

/* Demo: predict on last few samples, print directions */
void demo_direction(LSTMNetwork *network, Dataset *ds, int ohlc_predict_count);

#endif /* PREDICT_DIRECTION_H */
