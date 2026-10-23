#ifndef TYPES_H
#define TYPES_H

/* LSTM cell state for BPTT - standalone, no dependencies */

typedef struct {
    double *hidden_state;
    double *cell_state;
    double *forget_gate;
    double *input_gate;
    double *output_gate;
    double *candidate;
    double *concatenated;
} LSTMCellState;

/* Free per-timestep buffers and the array itself */
void lstm_free_cell_states(LSTMCellState *cell_states, int seq_len);

/* Free only per-timestep buffers (caller must free the array itself) */
void lstm_free_cell_state_buffers(LSTMCellState *cell_states, int seq_len);

#endif /* TYPES_H */
