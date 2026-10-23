#ifndef LSTM_H
#define LSTM_H

#include "utils.h"
#include "data.h"
#include "types.h"

/* ==================== Constants ==================== */
#define PREDICTION_THRESHOLD 0.5

/* ==================== LSTM Network ==================== */
typedef struct {
    int num_cells;
    int num_features;

    /* Weight matrices for each gate: [num_cells x num_cells] (recurrent)
     *                      and [num_cells x num_features] (input) */
    Matrix *weight_forget_recurrent, *weight_forget_input;
    double *bias_forget;

    Matrix *weight_input_recurrent, *weight_input_input;
    double *bias_input;

    Matrix *weight_output_recurrent, *weight_output_input;
    double *bias_output;

    Matrix *weight_candidate_recurrent, *weight_candidate_input;
    double *bias_candidate;

    /* Output layer: [1 x num_cells] and [1] */
    Matrix *weight_output_layer;
    double bias_output_layer;

    /* Gradients */
    Matrix *dweight_forget_recurrent, *dweight_forget_input;
    double *dbias_forget;

    Matrix *dweight_input_recurrent, *dweight_input_input;
    double *dbias_input;

    Matrix *dweight_output_recurrent, *dweight_output_input;
    double *dbias_output;

    Matrix *dweight_candidate_recurrent, *dweight_candidate_input;
    double *dbias_candidate;

    Matrix *dweight_output_layer;
    double dbias_output_layer;

    /* Momentum terms */
    Matrix *mweight_forget_recurrent, *mweight_forget_input;
    double *mbias_forget;
    Matrix *mweight_input_recurrent, *mweight_input_input;
    double *mbias_input;
    Matrix *mweight_output_recurrent, *mweight_output_input;
    double *mbias_output;
    Matrix *mweight_candidate_recurrent, *mweight_candidate_input;
    double *mbias_candidate;
    Matrix *mweight_output_layer;
    double mbias_output_layer;
} LSTMNetwork;

/* ==================== Functions ==================== */

/* Initialize network with Xavier/Glorot initialization */
LSTMNetwork *lstm_init(int num_cells, int num_features, unsigned int seed);

/* Free network */
void lstm_free(LSTMNetwork *net);

/* Forward pass through a single sequence */
/* Returns prediction probability [0, 1] */
/* Fills cell_states array (size = sequence_length) with intermediate values */
double lstm_forward(LSTMNetwork *net, const double *sequence,
                    LSTMCellState *cell_states, int seq_len);

/* Backward pass (BPTT) - computes gradients */
void lstm_backward(LSTMNetwork *net, const double *sequence,
                   const LSTMCellState *cell_states,
                   double prediction, double target, int seq_len);

/* Update weights with SGD + momentum (gradients divided by batch_size) */
void lstm_update(LSTMNetwork *net, double lr, double momentum, int batch_size);

/* Compute loss and accuracy on a dataset */
double lstm_compute_loss(LSTMNetwork *net, Dataset *ds);
double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds);

/* Predict probability for a single sequence */
double lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len);

#endif /* LSTM_H */
