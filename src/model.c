#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Load section identifiers ==================== */
#define SECTION_NONE 0
#define SECTION_NORMALIZATION 1
#define SECTION_LSTM_WEIGHTS 2
#define SECTION_OUTPUT_WEIGHTS 3

/* ==================== Load gate identifiers ==================== */
#define GATE_FORGET 0
#define GATE_INPUT 1
#define GATE_OUTPUT 2
#define GATE_CANDIDATE 3

/* ==================== Load matrix type identifiers ==================== */
#define MATRIX_TYPE_RECURRENT 0
#define MATRIX_TYPE_INPUT 1
#define MATRIX_TYPE_BIAS 2

/* ==================== Save ==================== */

int model_save(const char *filepath, LSTMNetwork *net,
               double epochs_trained, double best_train_loss,
               double best_test_accuracy, Dataset *norm_params) {
    FILE *f;
    int n, f_, combined, num_out, i, j;

    f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "[MODEL SAVE ERROR] Cannot open %s\n", filepath);
        return -1;
    }

    n = net->num_cells;
    f_ = net->num_features;
    num_out = net->num_outputs;
    combined = n + f_;

    fprintf(f, "# LSTM Model - Text Format\n");
    fprintf(f, "# num_cells: %d\n", n);
    fprintf(f, "# num_features: %d\n", f_);
    fprintf(f, "# num_outputs: %d\n", num_out);
    if (norm_params && norm_params->mode == MODE_DIRECTION)
        fprintf(f, "# predict_count: %d\n", num_out);
    else
        fprintf(f, "# predict_count: %d\n", num_out / OUTPUTS_PER_CANDLE);
    fprintf(f, "# mode: %s\n", (norm_params && norm_params->mode == MODE_DIRECTION) ? "direction" : "ohlc");
    fprintf(f, "# epochs_trained: %.0f\n", epochs_trained);
    fprintf(f, "# best_train_mse: %.6f\n", best_train_loss);
    fprintf(f, "# best_test_direction_accuracy: %.4f\n\n", best_test_accuracy);

    if (norm_params) {
        fprintf(f, "[normalization]\n");
        for (i = 0; i < norm_params->num_features; i++) {
            fprintf(f, "feature_name: %s | min: %.8f | max: %.8f\n",
                    norm_params->feat_names[i],
                    norm_params->feat_min[i],
                    norm_params->feat_max[i]);
        }
        fprintf(f, "label_min:");
        for (i = 0; i < norm_params->num_outputs; i++) {
            fprintf(f, " %.8f", norm_params->label_min[i]);
        }
        fprintf(f, "\n");
        fprintf(f, "label_max:");
        for (i = 0; i < norm_params->num_outputs; i++) {
            fprintf(f, " %.8f", norm_params->label_max[i]);
        }
        fprintf(f, "\n\n");
    }

    fprintf(f, "[lstm_weights]\n");

    #define SAVE_GATE(name) do { \
        fprintf(f, "# Gate: " #name "\n"); \
        fprintf(f, #name "_recurrent:\n"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, "  row[%d]:", i); \
            for (j = 0; j < combined; j++) { \
                fprintf(f, " %.6f", net->weight_##name##_recurrent->data[i * combined + j]); \
            } \
            fprintf(f, "\n"); \
        } \
        fprintf(f, #name "_input:\n"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, "  row[%d]:", i); \
            for (j = 0; j < f_; j++) { \
                fprintf(f, " %.6f", net->weight_##name##_input->data[i * f_ + j]); \
            } \
            fprintf(f, "\n"); \
        } \
        fprintf(f, #name "_bias:"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, " %.6f", net->bias_##name[i]); \
        } \
        fprintf(f, "\n\n"); \
    } while (0)

    SAVE_GATE(forget);
    SAVE_GATE(input);
    SAVE_GATE(output);
    SAVE_GATE(candidate);
    #undef SAVE_GATE

    fprintf(f, "[output_weights]\n");
    fprintf(f, "weight_output_layer:\n");
    for (i = 0; i < num_out; i++) {
        fprintf(f, "  row[%d]:", i);
        for (j = 0; j < n; j++) {
            fprintf(f, " %.6f", net->weight_output_layer->data[i * n + j]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "bias_output_layer:");
    for (i = 0; i < num_out; i++) {
        fprintf(f, " %.6f", net->bias_output_layer[i]);
    }
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

/* ==================== Load ==================== */

int model_load(const char *filepath, LSTMNetwork *net) {
    FILE *f;
    char line[8192];
    int n, f_, combined, num_out;
    int section;
    int gate;
    int matrix_type;
    double val;
    int parsed_row;
    int idx;
    int i;
    int dim;
    int rows_loaded = 0;
    int lines_processed = 0;
    int gate_set_count = 0;

    n = net->num_cells;
    f_ = net->num_features;
    num_out = net->num_outputs;
    combined = n + f_;

    f = fopen(filepath, "r");
    if (!f) return -1;

    section = SECTION_NONE;
    gate = -1;
    matrix_type = MATRIX_TYPE_RECURRENT;

    while (fgets(line, sizeof(line), f)) {
        utils_trim(line);
        lines_processed++;

        if (line[0] == 0) continue;

        /* Detect sections */
        if (strcmp(line, "[normalization]") == 0) { section = SECTION_NORMALIZATION; continue; }
        if (strcmp(line, "[lstm_weights]") == 0) { section = SECTION_LSTM_WEIGHTS; continue; }
        if (strcmp(line, "[output_weights]") == 0) { section = SECTION_OUTPUT_WEIGHTS; continue; }

        /* Skip metadata comments but parse gate comments */
        if (section == SECTION_NONE) {
            if (line[0] == '#') continue;
            continue;
        }
        
        /* Parse gate comments within lstm_weights section */
        if (section == SECTION_LSTM_WEIGHTS && line[0] == '#') {
            continue;
        }

        /* Parse LSTM weights */
        if (section == SECTION_LSTM_WEIGHTS) {
            if (strncmp(line, "forget_recurrent:", strlen("forget_recurrent:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "forget_input:", strlen("forget_input:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "forget_bias:", strlen("forget_bias:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "input_recurrent:", strlen("input_recurrent:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "input_input:", strlen("input_input:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "input_bias:", strlen("input_bias:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "output_recurrent:", strlen("output_recurrent:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "output_input:", strlen("output_input:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "output_bias:", strlen("output_bias:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "candidate_recurrent:", strlen("candidate_recurrent:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "candidate_input:", strlen("candidate_input:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "candidate_bias:", strlen("candidate_bias:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                continue; 
            }

            /* Row lines: "row[0]: v1 v2 v3 ..." */
            if (gate >= 0 && strncmp(line, "row[", strlen("row[")) == 0) {
                if (sscanf(line, "row[%d]:", &parsed_row) == 1) {
                    char *p = strchr(line, ':');
                    if (p) {
                        p++;
                        while (*p == ' ') p++;
                        dim = (matrix_type == MATRIX_TYPE_RECURRENT) ? combined : f_;
                        for (i = 0; i < dim && *p; i++) {
                            val = strtod(p, &p);
                            if (gate == GATE_FORGET) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_forget_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_forget_input->data[parsed_row * f_ + i] = val;
                            } else if (gate == GATE_INPUT) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_input_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_input_input->data[parsed_row * f_ + i] = val;
                            } else if (gate == GATE_OUTPUT) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_output_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_output_input->data[parsed_row * f_ + i] = val;
                            } else {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_candidate_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_candidate_input->data[parsed_row * f_ + i] = val;
                            }
                        }
                        rows_loaded++;
                    }
                }
                continue;
            }

            /* Bias lines: "forget_bias: v1 v2 v3 ..." */
            if (gate >= 0 && matrix_type == MATRIX_TYPE_BIAS) {
                char *p = line;
                while (*p && *p != ':') p++;
                if (*p == ':') p++;
                while (*p == ' ') p++;
                for (i = 0; i < n && *p; i++) {
                    val = strtod(p, &p);
                    if (gate == GATE_FORGET) net->bias_forget[i] = val;
                    else if (gate == GATE_INPUT) net->bias_input[i] = val;
                    else if (gate == GATE_OUTPUT) net->bias_output[i] = val;
                    else net->bias_candidate[i] = val;
                }
                continue;
            }
        }

        /* Parse output weights */
        if (section == SECTION_OUTPUT_WEIGHTS) {
            if (strncmp(line, "weight_output_layer:", strlen("weight_output_layer:")) == 0) continue;
            /* Row lines for output layer: "row[o]: w0 w1 ... wn" */
            if (strncmp(line, "row[", strlen("row[")) == 0) {
                if (sscanf(line, "row[%d]:", &parsed_row) == 1) {
                    char *p = strchr(line, ':');
                    if (p) {
                        p++;
                        while (*p == ' ') p++;
                        for (i = 0; i < n && *p; i++) {
                            val = strtod(p, &p);
                            net->weight_output_layer->data[parsed_row * n + i] = val;
                        }
                    }
                }
                continue;
            }
            if (strncmp(line, "bias_output_layer:", strlen("bias_output_layer:")) == 0) {
                char *p = line + strlen("bias_output_layer:");
                while (*p == ' ') p++;
                for (i = 0; i < num_out && *p; i++) {
                    net->bias_output_layer[i] = strtod(p, &p);
                }
                continue;
            }
        }
    }

    fclose(f);
    
    return 0;
}
