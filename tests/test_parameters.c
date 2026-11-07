#include "test_utils.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Test: parameters file loaded correctly ==================== */

int test_parameters_load(void) {
    ParamsFile pf;

    printf("[TEST] parameters file: values loaded correctly...");
    fflush(stdout);

    parse_parameters_file("../tests/cfg/test_params_load.cfg", &pf);

    if (!pf.has_cells || pf.cells != 64) { printf("\n  FAIL: cells\n"); return 0; }
    if (!pf.has_epochs || pf.epochs != 50) { printf("\n  FAIL: epochs\n"); return 0; }
    if (!pf.has_lr || fabs(pf.lr - 0.001) > 1e-9) { printf("\n  FAIL: lr\n"); return 0; }
    if (!pf.has_seq || pf.seq != 32) { printf("\n  FAIL: seq\n"); return 0; }
    if (!pf.has_mode || strcmp(pf.mode, "direction") != 0) { printf("\n  FAIL: mode\n"); return 0; }
    if (!pf.has_output || pf.output != 3) { printf("\n  FAIL: output\n"); return 0; }
    if (!pf.has_h || pf.h != 4) { printf("\n  FAIL: h\n"); return 0; }
    if (!pf.has_wd || fabs(pf.wd - 0.00001) > 1e-12) { printf("\n  FAIL: wd\n"); return 0; }
    if (!pf.has_patience || pf.patience != 10) { printf("\n  FAIL: patience\n"); return 0; }
    if (!pf.has_batch_size || pf.batch_size != 16) { printf("\n  FAIL: batch_size\n"); return 0; }
    if (!pf.has_momentum || fabs(pf.momentum - 0.95) > 1e-9) { printf("\n  FAIL: momentum\n"); return 0; }
    if (!pf.has_seed || pf.seed != 123) { printf("\n  FAIL: seed\n"); return 0; }
    if (!pf.has_clip_min || fabs(pf.clip_min - (-3.0)) > 1e-9) { printf("\n  FAIL: clip_min\n"); return 0; }
    if (!pf.has_clip_max || fabs(pf.clip_max - 3.0) > 1e-9) { printf("\n  FAIL: clip_max\n"); return 0; }
    if (!pf.has_train_ratio || fabs(pf.train_ratio - 0.9) > 1e-9) { printf("\n  FAIL: train_ratio\n"); return 0; }
    if (!pf.has_volume_return_clip || fabs(pf.volume_return_clip - 5.0) > 1e-9) { printf("\n  FAIL: volume_return_clip\n"); return 0; }
    if (!pf.has_forget_gate_bias_init || fabs(pf.forget_gate_bias_init - 0.5) > 1e-9) { printf("\n  FAIL: forget_gate_bias_init\n"); return 0; }
    if (!pf.has_prediction_threshold || fabs(pf.prediction_threshold - 0.6) > 1e-9) { printf("\n  FAIL: prediction_threshold\n"); return 0; }

    printf("\n  PASS: all 18 parameters loaded correctly\n");
    return 1;
}

/* ==================== Test: blank values use defaults ==================== */

int test_parameters_blank_defaults(void) {
    ParamsFile pf;

    printf("[TEST] parameters file: blank values use defaults...");
    fflush(stdout);

    parse_parameters_file("../tests/cfg/test_params_blank.cfg", &pf);

    /* Set values should be present */
    if (!pf.has_cells || pf.cells != 256) { printf("\n  FAIL: cells should be 256\n"); return 0; }
    if (!pf.has_lr || fabs(pf.lr - 0.01) > 1e-9) { printf("\n  FAIL: lr should be 0.01\n"); return 0; }
    if (!pf.has_momentum || fabs(pf.momentum - 0.8) > 1e-9) { printf("\n  FAIL: momentum should be 0.8\n"); return 0; }

    /* Blank values should NOT be set (has_* = 0) */
    if (pf.has_epochs) { printf("\n  FAIL: blank epochs should not be set\n"); return 0; }
    if (pf.has_seq) { printf("\n  FAIL: blank seq should not be set\n"); return 0; }
    if (pf.has_mode) { printf("\n  FAIL: blank mode should not be set\n"); return 0; }
    if (pf.has_batch_size) { printf("\n  FAIL: blank batch_size should not be set\n"); return 0; }

    printf("\n  PASS: blank values skipped, set values loaded\n");
    return 1;
}

/* ==================== Test: actual parameters.cfg loads ==================== */

int test_parameters_cfg_loads(void) {
    ParamsFile pf;

    printf("[TEST] parameters.cfg: project file loads correctly...");
    fflush(stdout);

    parse_parameters_file("../tests/cfg/parameters.cfg", &pf);

    /* Check default values from the test parameters.cfg */
    if (!pf.has_cells || pf.cells != 128) { printf("\n  FAIL: cells should be 128\n"); return 0; }
    if (!pf.has_epochs || pf.epochs != 100) { printf("\n  FAIL: epochs should be 100\n"); return 0; }
    if (!pf.has_lr || fabs(pf.lr - 0.005) > 1e-9) { printf("\n  FAIL: lr should be 0.005\n"); return 0; }
    if (!pf.has_seq || pf.seq != 64) { printf("\n  FAIL: seq should be 64\n"); return 0; }
    if (!pf.has_batch_size || pf.batch_size != 32) { printf("\n  FAIL: batch_size should be 32\n"); return 0; }
    if (!pf.has_momentum || fabs(pf.momentum - 0.9) > 1e-9) { printf("\n  FAIL: momentum should be 0.9\n"); return 0; }
    if (!pf.has_seed || pf.seed != 42) { printf("\n  FAIL: seed should be 42\n"); return 0; }
    if (!pf.has_clip_min || fabs(pf.clip_min - (-5.0)) > 1e-9) { printf("\n  FAIL: clip_min should be -5.0\n"); return 0; }
    if (!pf.has_clip_max || fabs(pf.clip_max - 5.0) > 1e-9) { printf("\n  FAIL: clip_max should be 5.0\n"); return 0; }
    if (!pf.has_train_ratio || fabs(pf.train_ratio - 0.8) > 1e-9) { printf("\n  FAIL: train_ratio should be 0.8\n"); return 0; }
    if (!pf.has_forget_gate_bias_init || fabs(pf.forget_gate_bias_init - 1.0) > 1e-9) { printf("\n  FAIL: forget_gate_bias_init should be 1.0\n"); return 0; }

    /* Blank values should not be set */
    if (pf.has_datetime) { printf("\n  FAIL: datetime should be blank\n"); return 0; }
    if (pf.has_sdata) { printf("\n  FAIL: sdata should be blank\n"); return 0; }
    if (pf.has_model) { printf("\n  FAIL: model should be blank\n"); return 0; }

    printf("\n  PASS: parameters.cfg loaded with correct defaults\n");
    return 1;
}
