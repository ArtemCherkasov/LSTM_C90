#ifndef MODEL_H
#define MODEL_H

#include "lstm.h"
#include "data.h"
#include "types.h"

/* ==================== Model Persistence ==================== */

/* Save network to human-readable text file */
int model_save(const char *filepath, LSTMNetwork *net,
               double epochs_trained, double best_train_loss,
               double best_test_accuracy, Dataset *norm_params);

/* Load network from human-readable text file */
int model_load(const char *filepath, LSTMNetwork *net);

#endif /* MODEL_H */
