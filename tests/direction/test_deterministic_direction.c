#include "../test_utils.h"
#include <stdio.h>
#include <math.h>

/* ==================== Deterministic forward+BPTT for direction (sigmoid output) ==================== */
/* 1 cell, 1 feature, 1 output, 2 timesteps.
 * All gates: w_h=0.1 (recurrent), w_x=0.3 (input), bias=0.
 * Output layer: w=0.5, b=0. Sigmoid activation.
 * Inputs: x[0]=0.7, x[1]=0.4. Targets: t[0]=0.0, t[1]=1.0.
 * Expected values computed by independent scalar math, compared against LSTM code. */

#define DDIR_W_H    0.1
#define DDIR_W_X    0.3
#define DDIR_W_OUT  0.5
#define DDIR_X0     0.7
#define DDIR_X1     0.4
#define DDIR_TOL    1e-8

static void set_simple_weights_dir(LSTMNetwork *net) {
    net->weight_forget_recurrent->data[0] = DDIR_W_H;
    net->weight_forget_recurrent->data[1] = DDIR_W_X;
    net->bias_forget[0] = 0.0;
    net->weight_input_recurrent->data[0] = DDIR_W_H;
    net->weight_input_recurrent->data[1] = DDIR_W_X;
    net->bias_input[0] = 0.0;
    net->weight_output_recurrent->data[0] = DDIR_W_H;
    net->weight_output_recurrent->data[1] = DDIR_W_X;
    net->bias_output[0] = 0.0;
    net->weight_candidate_recurrent->data[0] = DDIR_W_H;
    net->weight_candidate_recurrent->data[1] = DDIR_W_X;
    net->bias_candidate[0] = 0.0;
    net->weight_output_layer->data[0] = DDIR_W_OUT;
    net->bias_output_layer[0] = 0.0;
    net->output_activation = ACTIVATION_SIGMOID;
}

int test_deterministic_direction(void) {
    LSTMNetwork *net;
    LSTMCellState *cs;
    double sequence[2] = {DDIR_X0, DDIR_X1};
    double preds[2];
    double targets[2] = {0.0, 1.0};
    int fail = 0;

    /* Independent scalar forward — LSTM gates identical to OHLC, only output differs */
    double pre0 = DDIR_W_H * 0.0 + DDIR_W_X * DDIR_X0;
    double f0 = 1.0 / (1.0 + exp(-pre0));
    double i0 = f0, g0 = tanh(pre0), o0 = f0;
    double c0 = i0 * g0;
    double h0 = o0 * tanh(c0);
    double z0 = DDIR_W_OUT * h0;           /* pre-activation */
    double y0 = 1.0 / (1.0 + exp(-z0));   /* sigmoid output */

    double pre1 = DDIR_W_H * h0 + DDIR_W_X * DDIR_X1;
    double f1 = 1.0 / (1.0 + exp(-pre1));
    double i1 = f1, g1 = tanh(pre1), o1 = f1;
    double c1 = f1 * c0 + i1 * g1;
    double h1 = o1 * tanh(c1);
    double z1 = DDIR_W_OUT * h1;
    double y1 = 1.0 / (1.0 + exp(-z1));

    printf("[TEST] Deterministic direction forward+BPTT (1 cell, 2 steps, sigmoid)...");
    fflush(stdout);

    net = lstm_init(1, 1, 1, 42);
    set_simple_weights_dir(net);
    cs = alloc_cell_states(2, 1, 2);
    lstm_forward(net, sequence, cs, 2, preds);

    if (fabs(preds[0] - y0) > DDIR_TOL) { printf("\n  FAIL: preds[0]=%.10f exp=%.10f", preds[0], y0); fail = 1; }
    if (fabs(preds[1] - y1) > DDIR_TOL) { printf("\n  FAIL: preds[1]=%.10f exp=%.10f", preds[1], y1); fail = 1; }
    if (fabs(cs[0].hidden_state[0] - h0) > DDIR_TOL) { printf("\n  FAIL: h0=%.10f exp=%.10f", cs[0].hidden_state[0], h0); fail = 1; }
    if (fabs(cs[1].hidden_state[0] - h1) > DDIR_TOL) { printf("\n  FAIL: h1=%.10f exp=%.10f", cs[1].hidden_state[0], h1); fail = 1; }

    /* Independent scalar BPTT — sigmoid derivative: dL/dz = (y-t) * y*(1-y) */
    {
        double dLdy1_raw = y1 - targets[1];
        double dLdz1 = dLdy1_raw * y1 * (1.0 - y1);  /* sigmoid derivative */
        double dLdh1 = dLdz1 * DDIR_W_OUT;
        double tc1 = tanh(c1);
        double dLdc1 = dLdh1 * o1 * (1.0 - tc1 * tc1);
        double dLdo1 = dLdh1 * tc1 * o1 * (1.0 - o1);
        double dLdi1 = dLdc1 * g1 * i1 * (1.0 - i1);
        double dLdg1 = dLdc1 * i1 * (1.0 - g1 * g1);
        double dLdf1 = dLdc1 * c0 * f1 * (1.0 - f1);

        double dconcat0 = DDIR_W_H * (dLdf1 + dLdi1 + dLdo1 + dLdg1);

        double dLdy0_raw = y0 - targets[0];
        double dLdz0 = dLdy0_raw * y0 * (1.0 - y0);
        double dLdh0 = dLdz0 * DDIR_W_OUT + dconcat0;
        double tc0 = tanh(c0);
        double dLdc0 = dLdh0 * o0 * (1.0 - tc0 * tc0) + dLdc1 * f1;
        double dLdo0 = dLdh0 * tc0 * o0 * (1.0 - o0);
        double dLdi0 = dLdc0 * g0 * i0 * (1.0 - i0);
        double dLdg0 = dLdc0 * i0 * (1.0 - g0 * g0);
        double dLdf0 = dLdc0 * 0.0 * f0 * (1.0 - f0);

        double exp_dw_out  = dLdz1 * h1 + dLdz0 * h0;
        double exp_db_out  = dLdz1 + dLdz0;
        double exp_dwi_x   = dLdi1 * DDIR_X1 + dLdi0 * DDIR_X0;
        double exp_dwo_x   = dLdo1 * DDIR_X1 + dLdo0 * DDIR_X0;
        double exp_dwc_x   = dLdg1 * DDIR_X1 + dLdg0 * DDIR_X0;
        double exp_dwf_x   = dLdf1 * DDIR_X1 + dLdf0 * DDIR_X0;
        double exp_dwi_h   = dLdi1 * h0 + dLdi0 * 0.0;
        double exp_dbi     = dLdi1 + dLdi0;
        double exp_dbo     = dLdo1 + dLdo0;
        double exp_dbc     = dLdg1 + dLdg0;
        double exp_dbf     = dLdf1 + dLdf0;

        lstm_zero_gradients(net);
        lstm_backward(net, sequence, cs, preds, targets, 2);

        if (fabs(net->dweight_output_layer->data[0] - exp_dw_out) > DDIR_TOL) { printf("\n  FAIL: dw_out=%.10f exp=%.10f", net->dweight_output_layer->data[0], exp_dw_out); fail = 1; }
        if (fabs(net->dbias_output_layer[0] - exp_db_out) > DDIR_TOL) { printf("\n  FAIL: db_out=%.10f exp=%.10f", net->dbias_output_layer[0], exp_db_out); fail = 1; }
        if (fabs(net->dweight_input_recurrent->data[1] - exp_dwi_x) > DDIR_TOL) { printf("\n  FAIL: dwi_x=%.10f exp=%.10f", net->dweight_input_recurrent->data[1], exp_dwi_x); fail = 1; }
        if (fabs(net->dweight_output_recurrent->data[1] - exp_dwo_x) > DDIR_TOL) { printf("\n  FAIL: dwo_x=%.10f exp=%.10f", net->dweight_output_recurrent->data[1], exp_dwo_x); fail = 1; }
        if (fabs(net->dweight_candidate_recurrent->data[1] - exp_dwc_x) > DDIR_TOL) { printf("\n  FAIL: dwc_x=%.10f exp=%.10f", net->dweight_candidate_recurrent->data[1], exp_dwc_x); fail = 1; }
        if (fabs(net->dweight_forget_recurrent->data[1] - exp_dwf_x) > DDIR_TOL) { printf("\n  FAIL: dwf_x=%.10f exp=%.10f", net->dweight_forget_recurrent->data[1], exp_dwf_x); fail = 1; }
        if (fabs(net->dweight_input_recurrent->data[0] - exp_dwi_h) > DDIR_TOL) { printf("\n  FAIL: dwi_h=%.10f exp=%.10f", net->dweight_input_recurrent->data[0], exp_dwi_h); fail = 1; }
        if (fabs(net->dbias_input[0] - exp_dbi) > DDIR_TOL) { printf("\n  FAIL: dbi=%.10f exp=%.10f", net->dbias_input[0], exp_dbi); fail = 1; }
        if (fabs(net->dbias_output[0] - exp_dbo) > DDIR_TOL) { printf("\n  FAIL: dbo=%.10f exp=%.10f", net->dbias_output[0], exp_dbo); fail = 1; }
        if (fabs(net->dbias_candidate[0] - exp_dbc) > DDIR_TOL) { printf("\n  FAIL: dbc=%.10f exp=%.10f", net->dbias_candidate[0], exp_dbc); fail = 1; }
        if (fabs(net->dbias_forget[0] - exp_dbf) > DDIR_TOL) { printf("\n  FAIL: dbf=%.10f exp=%.10f", net->dbias_forget[0], exp_dbf); fail = 1; }
    }

    if (!fail) printf("\n  y0=%.8f y1=%.8f  PASS: forward+BPTT match scalar\n", y0, y1);

    lstm_free_cell_states(cs, 2);
    lstm_free(net);
    return !fail;
}
