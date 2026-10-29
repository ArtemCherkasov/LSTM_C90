#include "lstm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Initialization ==================== */

LSTMNetwork *lstm_init(int num_cells, int num_features, int num_outputs, unsigned int seed) {
    LSTMNetwork *net;
    int i;
    int fan_in;
    double scale;

    utils_srand(seed);

    net = (LSTMNetwork *)utils_safe_malloc(sizeof(LSTMNetwork));
    net->num_cells = num_cells;
    net->num_features = num_features;
    net->num_outputs = num_outputs;
    net->output_activation = ACTIVATION_LINEAR;
    fan_in = num_cells + num_features;
    scale = sqrt(2.0 / ((double)(fan_in + num_cells)));

    #define GATE_ALLOC(name) do { \
        net->weight_##name##_recurrent = matrix_alloc(num_cells, fan_in); \
        net->weight_##name##_input = matrix_alloc(num_cells, num_features); \
        net->bias_##name = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double)); \
        net->mweight_##name##_recurrent = matrix_alloc(num_cells, fan_in); \
        net->mweight_##name##_input = matrix_alloc(num_cells, num_features); \
        net->mbias_##name = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double)); \
        net->dweight_##name##_recurrent = matrix_alloc(num_cells, fan_in); \
        net->dweight_##name##_input = matrix_alloc(num_cells, num_features); \
        net->dbias_##name = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double)); \
        for (i = 0; i < num_cells * fan_in; i++) { \
            net->weight_##name##_recurrent->data[i] = utils_gaussian_random(0.0, scale); \
        } \
        for (i = 0; i < num_cells * num_features; i++) { \
            net->weight_##name##_input->data[i] = utils_gaussian_random(0.0, scale); \
        } \
    } while (0)

    GATE_ALLOC(forget);
    GATE_ALLOC(input);
    GATE_ALLOC(output);
    GATE_ALLOC(candidate);
    #undef GATE_ALLOC

    net->weight_output_layer = matrix_alloc(num_outputs, num_cells);
    net->mweight_output_layer = matrix_alloc(num_outputs, num_cells);
    net->dweight_output_layer = matrix_alloc(num_outputs, num_cells);
    for (i = 0; i < num_outputs * num_cells; i++) {
        net->weight_output_layer->data[i] = utils_gaussian_random(0.0, scale);
    }
    net->bias_output_layer = (double *)utils_safe_calloc((size_t)num_outputs, sizeof(double));
    net->mbias_output_layer = (double *)utils_safe_calloc((size_t)num_outputs, sizeof(double));
    net->dbias_output_layer = (double *)utils_safe_calloc((size_t)num_outputs, sizeof(double));

    return net;
}

void lstm_free(LSTMNetwork *net) {
    if (!net) return;

    #define GATE_FREE(name) do { \
        matrix_free(net->weight_##name##_recurrent); \
        matrix_free(net->weight_##name##_input); \
        free(net->bias_##name); \
        matrix_free(net->mweight_##name##_recurrent); \
        matrix_free(net->mweight_##name##_input); \
        free(net->mbias_##name); \
        matrix_free(net->dweight_##name##_recurrent); \
        matrix_free(net->dweight_##name##_input); \
        free(net->dbias_##name); \
    } while (0)

    GATE_FREE(forget);
    GATE_FREE(input);
    GATE_FREE(output);
    GATE_FREE(candidate);
    #undef GATE_FREE

    matrix_free(net->weight_output_layer);
    matrix_free(net->mweight_output_layer);
    matrix_free(net->dweight_output_layer);
    free(net->bias_output_layer);
    free(net->mbias_output_layer);
    free(net->dbias_output_layer);
    free(net);
}

void lstm_free_cell_states(LSTMCellState *cell_states, int seq_len) {
    int t;
    if (!cell_states) return;
    for (t = 0; t < seq_len; t++) {
        free(cell_states[t].hidden_state);
        free(cell_states[t].cell_state);
        free(cell_states[t].forget_gate);
        free(cell_states[t].input_gate);
        free(cell_states[t].output_gate);
        free(cell_states[t].candidate);
        free(cell_states[t].concatenated);
    }
    free(cell_states);
}

void lstm_free_cell_state_buffers(LSTMCellState *cell_states, int seq_len) {
    int t;
    if (!cell_states) return;
    for (t = 0; t < seq_len; t++) {
        free(cell_states[t].hidden_state);
        free(cell_states[t].cell_state);
        free(cell_states[t].forget_gate);
        free(cell_states[t].input_gate);
        free(cell_states[t].output_gate);
        free(cell_states[t].candidate);
        free(cell_states[t].concatenated);
    }
}

/* ==================== Forward Pass ==================== */

void lstm_forward(LSTMNetwork *net, const double *sequence,
                  LSTMCellState *cell_states, int seq_len,
                  double *outputs) {
    int t, c, r, f, n, combined, o;
    double *hidden_state, *cell_state;
    double *forget_gate, *input_gate, *output_gate;
    double *candidate_vec, *concatenated;
    double *Wf, *Wi, *Wo, *Wc;
    double *bf, *bi, *bo, *bc;
    double *W_out;
    double sum, tanh_c;

    n = net->num_cells;
    f = net->num_features;
    combined = n + f;

    hidden_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    cell_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    forget_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    input_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    output_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    candidate_vec = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));

    Wf = net->weight_forget_recurrent->data;
    Wi = net->weight_input_recurrent->data;
    Wo = net->weight_output_recurrent->data;
    Wc = net->weight_candidate_recurrent->data;
    bf = net->bias_forget;
    bi = net->bias_input;
    bo = net->bias_output;
    bc = net->bias_candidate;

    W_out = net->weight_output_layer->data;

    for (t = 0; t < seq_len; t++) {
        for (c = 0; c < n; c++) {
            concatenated[c] = hidden_state[c];
        }
        for (c = 0; c < f; c++) {
            concatenated[n + c] = sequence[t * f + c];
        }

        /* Forget gate */
        for (c = 0; c < n; c++) {
            sum = bf[c];
            for (r = 0; r < combined; r++) {
                sum += Wf[c * combined + r] * concatenated[r];
            }
            forget_gate[c] = utils_sigmoid(sum);
        }

        /* Input gate */
        for (c = 0; c < n; c++) {
            sum = bi[c];
            for (r = 0; r < combined; r++) {
                sum += Wi[c * combined + r] * concatenated[r];
            }
            input_gate[c] = utils_sigmoid(sum);
        }

        /* Candidate */
        for (c = 0; c < n; c++) {
            sum = bc[c];
            for (r = 0; r < combined; r++) {
                sum += Wc[c * combined + r] * concatenated[r];
            }
            candidate_vec[c] = utils_tanh_act(sum);
        }

        /* Cell state */
        for (c = 0; c < n; c++) {
            cell_state[c] = forget_gate[c] * cell_state[c] +
                            input_gate[c] * candidate_vec[c];
        }

        /* Output gate */
        for (c = 0; c < n; c++) {
            sum = bo[c];
            for (r = 0; r < combined; r++) {
                sum += Wo[c * combined + r] * concatenated[r];
            }
            output_gate[c] = utils_sigmoid(sum);
        }

        /* Hidden state */
        for (c = 0; c < n; c++) {
            tanh_c = utils_tanh_act(cell_state[c]);
            hidden_state[c] = output_gate[c] * tanh_c;
        }

        /* Store state */
        for (c = 0; c < n; c++) {
            cell_states[t].hidden_state[c] = hidden_state[c];
            cell_states[t].cell_state[c] = cell_state[c];
            cell_states[t].forget_gate[c] = forget_gate[c];
            cell_states[t].input_gate[c] = input_gate[c];
            cell_states[t].output_gate[c] = output_gate[c];
            cell_states[t].candidate[c] = candidate_vec[c];
        }
        for (c = 0; c < combined; c++) {
            cell_states[t].concatenated[c] = concatenated[c];
        }

        /* Output layer at this timestep (many-to-many) */
        for (o = 0; o < net->num_outputs; o++) {
            sum = net->bias_output_layer[o];
            for (c = 0; c < n; c++) {
                sum += W_out[o * n + c] * hidden_state[c];
            }
            if (net->output_activation == ACTIVATION_SIGMOID)
                outputs[t * net->num_outputs + o] = utils_sigmoid(sum);
            else
                outputs[t * net->num_outputs + o] = sum;
        }
    }

    free(hidden_state);
    free(cell_state);
    free(forget_gate);
    free(input_gate);
    free(output_gate);
    free(candidate_vec);
    free(concatenated);
}

/* ==================== Gradient Zeroing ==================== */

void lstm_zero_gradients(LSTMNetwork *net) {
    int c;
    int num_cells = net->num_cells;
    
    #define GATE_ZERO(name) do { \
        for (c = 0; c < (net->dweight_##name##_recurrent)->rows * \
                            (net->dweight_##name##_recurrent)->cols; c++) \
            (net->dweight_##name##_recurrent)->data[c] = 0.0; \
        for (c = 0; c < (net->dweight_##name##_input)->rows * \
                            (net->dweight_##name##_input)->cols; c++) \
            (net->dweight_##name##_input)->data[c] = 0.0; \
        for (c = 0; c < (size_t)num_cells; c++) (net->dbias_##name)[c] = 0.0; \
    } while (0)

    GATE_ZERO(forget);
    GATE_ZERO(input);
    GATE_ZERO(output);
    GATE_ZERO(candidate);

    for (c = 0; c < (net->dweight_output_layer)->rows * \
                        (net->dweight_output_layer)->cols; c++) {
        net->dweight_output_layer->data[c] = 0.0;
    }
    for (c = 0; c < net->num_outputs; c++) {
        net->dbias_output_layer[c] = 0.0;
    }
    
    #undef GATE_ZERO
}

/* ==================== Backward Pass (BPTT) ==================== */

void lstm_backward(LSTMNetwork *net, const double *sequence,
                   const LSTMCellState *cell_states,
                   const double *predictions, const double *targets,
                   int seq_len) {
    int t, c, r, o, n, f, combined;
    double *dhidden, *dcell;
    double *W_out;
    const LSTMCellState *cs;
    const double *c_prev;
    const double *forget_g, *input_g, *output_g, *cand, *concat;
    double *d_output_gate, *d_cell_from_hidden;
    double *d_forget_gate, *d_input_gate, *d_candidate;
    double *dconcat;
    double tanh_c;
    double *Wf, *Wi, *Wo, *Wc;
    int c_idx, r_idx;

    n = net->num_cells;
    f = net->num_features;
    combined = n + f;

    W_out = net->weight_output_layer->data;

    dhidden = (double *)utils_safe_calloc((size_t)n, sizeof(double));
    dcell = (double *)utils_safe_calloc((size_t)n, sizeof(double));

    Wf = net->weight_forget_recurrent->data;
    Wi = net->weight_input_recurrent->data;
    Wo = net->weight_output_recurrent->data;
    Wc = net->weight_candidate_recurrent->data;

    for (t = seq_len - 1; t >= 0; t--) {
        double d_logit_o;
        int out_base = t * net->num_outputs;

        cs = &cell_states[t];
        c_prev = (t > 0) ? cell_states[t - 1].cell_state : NULL;
        forget_g = cs->forget_gate;
        input_g = cs->input_gate;
        output_g = cs->output_gate;
        cand = cs->candidate;
        concat = cs->concatenated;

        /* Add output layer gradient at this timestep (many-to-many) */
        for (o = 0; o < net->num_outputs; o++) {
            double pred_o = predictions[out_base + o];
            d_logit_o = pred_o - targets[out_base + o];
            if (net->output_activation == ACTIVATION_SIGMOID)
                d_logit_o *= pred_o * (1.0 - pred_o);
            for (c = 0; c < n; c++) {
                net->dweight_output_layer->data[o * n + c] += d_logit_o * cs->hidden_state[c];
            }
            net->dbias_output_layer[o] += d_logit_o;
            for (c = 0; c < n; c++) {
                dhidden[c] += d_logit_o * W_out[o * n + c];
            }
        }

        d_output_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        d_cell_from_hidden = (double *)utils_safe_calloc((size_t)n, sizeof(double));

        for (c = 0; c < n; c++) {
            tanh_c = utils_tanh_act(cs->cell_state[c]);
            if (tanh_c > 0.9999) tanh_c = 0.9999;
            if (tanh_c < -0.9999) tanh_c = -0.9999;
            d_output_gate[c] = dhidden[c] * tanh_c * output_g[c] * (1.0 - output_g[c]);
            d_cell_from_hidden[c] = dhidden[c] * output_g[c] * (1.0 - tanh_c * tanh_c);
        }

        for (c = 0; c < n; c++) {
            dcell[c] += d_cell_from_hidden[c];
        }

        d_forget_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        d_input_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        d_candidate = (double *)utils_safe_calloc((size_t)n, sizeof(double));

        for (c = 0; c < n; c++) {
            double fg = forget_g[c];
            double ig = input_g[c];
            double cg = cand[c];
            if (fg > 0.9999) fg = 0.9999;
            if (fg < 0.0001) fg = 0.0001;
            if (ig > 0.9999) ig = 0.9999;
            if (ig < 0.0001) ig = 0.0001;
            if (cg > 0.9999) cg = 0.9999;
            if (cg < -0.9999) cg = -0.9999;

            d_forget_gate[c] = dcell[c] *
                               ((c_prev ? c_prev[c] : 0.0)) *
                               fg * (1.0 - fg);
            d_input_gate[c] = dcell[c] * cand[c] *
                              ig * (1.0 - ig);
            d_candidate[c] = dcell[c] * input_g[c] *
                             (1.0 - cg * cg);
        }

        /* Accumulate gradients */
        for (r = 0; r < n; r++) {
            for (c_idx = 0; c_idx < combined; c_idx++) {
                net->dweight_forget_recurrent->data[r * combined + c_idx] +=
                    d_forget_gate[r] * concat[c_idx];
                net->dweight_input_recurrent->data[r * combined + c_idx] +=
                    d_input_gate[r] * concat[c_idx];
                net->dweight_output_recurrent->data[r * combined + c_idx] +=
                    d_output_gate[r] * concat[c_idx];
                net->dweight_candidate_recurrent->data[r * combined + c_idx] +=
                    d_candidate[r] * concat[c_idx];
            }
            net->dbias_forget[r] += d_forget_gate[r];
            net->dbias_input[r] += d_input_gate[r];
            net->dbias_output[r] += d_output_gate[r];
            net->dbias_candidate[r] += d_candidate[r];
        }

        dconcat = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
        for (c_idx = 0; c_idx < combined; c_idx++) {
            for (r_idx = 0; r_idx < n; r_idx++) {
                dconcat[c_idx] += Wf[r_idx * combined + c_idx] * d_forget_gate[r_idx];
                dconcat[c_idx] += Wi[r_idx * combined + c_idx] * d_input_gate[r_idx];
                dconcat[c_idx] += Wo[r_idx * combined + c_idx] * d_output_gate[r_idx];
                dconcat[c_idx] += Wc[r_idx * combined + c_idx] * d_candidate[r_idx];
            }
        }

        if (t > 0) {
            for (c = 0; c < n; c++) {
                dhidden[c] = dconcat[c];
            }
        }

        /* Propagate dcell to previous timestep: dL/dc_{t-1} = dL/dc_t * f_t */
        if (t > 0) {
            for (c = 0; c < n; c++) {
                dcell[c] = dcell[c] * forget_g[c];
            }
        }

        free(d_output_gate);
        free(d_cell_from_hidden);
        free(d_forget_gate);
        free(d_input_gate);
        free(d_candidate);
        free(dconcat);
    }
    free(dhidden);
    free(dcell);
}

/* ==================== Update Weights ==================== */

void lstm_update(LSTMNetwork *net, double lr, double momentum, int batch_size) {
    int c, ui;
    int num_cells = net->num_cells;
    double weight_decay = 1e-6;
    double inv_batch = 1.0 / (double)batch_size;

    /* Normalize gradients by batch size and clip */
    #define CLIP_GRAD(name) do { \
        for (ui = 0; ui < (net->dweight_##name##_recurrent)->rows * \
                            (net->dweight_##name##_recurrent)->cols; ui++) { \
            net->dweight_##name##_recurrent->data[ui] = utils_clip( \
                net->dweight_##name##_recurrent->data[ui] * inv_batch, CLIP_MIN, CLIP_MAX); \
        } \
        for (ui = 0; ui < (net->dweight_##name##_input)->rows * \
                            (net->dweight_##name##_input)->cols; ui++) { \
            net->dweight_##name##_input->data[ui] = utils_clip( \
                net->dweight_##name##_input->data[ui] * inv_batch, CLIP_MIN, CLIP_MAX); \
        } \
        for (ui = 0; ui < (size_t)num_cells; ui++) { \
            net->dbias_##name[ui] = utils_clip(net->dbias_##name[ui] * inv_batch, CLIP_MIN, CLIP_MAX); \
        } \
    } while (0)

    CLIP_GRAD(forget);
    CLIP_GRAD(input);
    CLIP_GRAD(output);
    CLIP_GRAD(candidate);

    for (ui = 0; ui < (net->dweight_output_layer)->rows * \
                        (net->dweight_output_layer)->cols; ui++) {
        net->dweight_output_layer->data[ui] = utils_clip(
            net->dweight_output_layer->data[ui] * inv_batch, CLIP_MIN, CLIP_MAX);
    }
    for (ui = 0; ui < net->num_outputs; ui++) {
        net->dbias_output_layer[ui] = utils_clip(
            net->dbias_output_layer[ui] * inv_batch, CLIP_MIN, CLIP_MAX);
    }

    #undef CLIP_GRAD

    #define GATE_UPDATE(name) do { \
        for (ui = 0; ui < (net->dweight_##name##_recurrent)->rows * \
                            (net->dweight_##name##_recurrent)->cols; ui++) { \
            net->mweight_##name##_recurrent->data[ui] = \
                momentum * net->mweight_##name##_recurrent->data[ui] - \
                lr * (net->dweight_##name##_recurrent->data[ui] + weight_decay * net->weight_##name##_recurrent->data[ui]); \
            net->weight_##name##_recurrent->data[ui] += \
                net->mweight_##name##_recurrent->data[ui]; \
        } \
        for (ui = 0; ui < (net->dweight_##name##_input)->rows * \
                            (net->dweight_##name##_input)->cols; ui++) { \
            net->mweight_##name##_input->data[ui] = \
                momentum * net->mweight_##name##_input->data[ui] - \
                lr * (net->dweight_##name##_input->data[ui] + weight_decay * net->weight_##name##_input->data[ui]); \
            net->weight_##name##_input->data[ui] += \
                net->mweight_##name##_input->data[ui]; \
        } \
        for (ui = 0; ui < (size_t)num_cells; ui++) { \
            net->mbias_##name[ui] = \
                momentum * net->mbias_##name[ui] - \
                lr * net->dbias_##name[ui]; \
            net->bias_##name[ui] += net->mbias_##name[ui]; \
        } \
    } while (0)

    GATE_UPDATE(forget);
    GATE_UPDATE(input);
    GATE_UPDATE(output);
    GATE_UPDATE(candidate);
    #undef GATE_UPDATE

    /* Output layer update */
    for (ui = 0; ui < net->num_outputs * num_cells; ui++) {
        net->mweight_output_layer->data[ui] = momentum * net->mweight_output_layer->data[ui]
            - lr * (net->dweight_output_layer->data[ui] + weight_decay * net->weight_output_layer->data[ui]);
        net->weight_output_layer->data[ui] += net->mweight_output_layer->data[ui];
    }
    for (c = 0; c < net->num_outputs; c++) {
        net->mbias_output_layer[c] = momentum * net->mbias_output_layer[c] - lr * net->dbias_output_layer[c];
        net->bias_output_layer[c] += net->mbias_output_layer[c];
    }
}

/* ==================== Loss & Accuracy ==================== */

double lstm_compute_loss(LSTMNetwork *net, Dataset *ds) {
    int i, t, o;
    double total_loss = 0.0;
    double *preds;
    double error;
    LSTMCellState *cell_states;
    int n = net->num_cells;
    int combined = n + net->num_features;

    cell_states = (LSTMCellState *)utils_safe_calloc(
        (size_t)ds->sequence_length, sizeof(LSTMCellState));

    for (t = 0; t < ds->sequence_length; t++) {
        cell_states[t].hidden_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].cell_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].forget_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].input_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].output_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].candidate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }

    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * ds->sequence_length), sizeof(double));

    for (i = 0; i < ds->num_samples; i++) {
        lstm_forward(net, ds->features[i], cell_states, ds->sequence_length, preds);
        for (t = 0; t < ds->sequence_length; t++) {
            for (o = 0; o < net->num_outputs; o++) {
                error = preds[t * net->num_outputs + o] - ds->labels[i][t * net->num_outputs + o];
                total_loss += 0.5 * error * error;
            }
        }
    }

    free(preds);
    lstm_free_cell_states(cell_states, ds->sequence_length);
    return total_loss / ((double)ds->num_samples * (double)ds->sequence_length * (double)net->num_outputs);
}

double lstm_compute_accuracy(LSTMNetwork *net, Dataset *ds) {
    int i, t, o, correct = 0;
    double *preds;
    LSTMCellState *cell_states;
    int n = net->num_cells;
    int combined = n + net->num_features;

    cell_states = (LSTMCellState *)utils_safe_calloc(
        (size_t)ds->sequence_length, sizeof(LSTMCellState));

    for (t = 0; t < ds->sequence_length; t++) {
        cell_states[t].hidden_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].cell_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].forget_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].input_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].output_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].candidate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }

    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * ds->sequence_length), sizeof(double));

    for (i = 0; i < ds->num_samples; i++) {
        lstm_forward(net, ds->features[i], cell_states, ds->sequence_length, preds);
        for (t = 0; t < ds->sequence_length; t++) {
            for (o = 0; o < net->num_outputs; o++) {
                int predicted = (preds[t * net->num_outputs + o] > 0.0) ? 1 : 0;
                int actual = (ds->labels[i][t * net->num_outputs + o] > 0.0) ? 1 : 0;
                if (predicted == actual) correct++;
            }
        }
    }

    free(preds);
    lstm_free_cell_states(cell_states, ds->sequence_length);
    return (double)correct / ((double)ds->num_samples * (double)ds->sequence_length * (double)net->num_outputs);
}

/* ==================== Predict ==================== */

void lstm_predict(LSTMNetwork *net, const double *sequence, int seq_len,
                  double *outputs) {
    LSTMCellState *cell_states;
    int t;
    int n = net->num_cells;
    int combined = n + net->num_features;

    cell_states = (LSTMCellState *)utils_safe_calloc(
        (size_t)seq_len, sizeof(LSTMCellState));

    for (t = 0; t < seq_len; t++) {
        cell_states[t].hidden_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].cell_state = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].forget_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].input_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].output_gate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].candidate = (double *)utils_safe_calloc((size_t)n, sizeof(double));
        cell_states[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }

    lstm_forward(net, sequence, cell_states, seq_len, outputs);
    lstm_free_cell_states(cell_states, seq_len);
}
