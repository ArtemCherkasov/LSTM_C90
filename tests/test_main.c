#include "test_utils.h"
#include <stdio.h>

int main(void) {
    int passed = 0;
    int total = 0;

    printf("========================================\n");
    printf("LSTM Network Tests (C90, multi-output)\n");
    printf("========================================\n\n");

    utils_srand(12345);

    /* ===== Common LSTM tests ===== */
    total++; passed += test_forward_pass();
    printf("\n");

    total++; passed += test_concatenated_storage();
    printf("\n");

    total++; passed += test_state_propagation();
    printf("\n");

    total++; passed += test_backward_pass();
    printf("\n");

    total++; passed += test_last_timestep_grad();
    printf("\n");

    total++; passed += test_output_grad_accumulation();
    printf("\n");

    total++; passed += test_update();
    printf("\n");

    total++; passed += test_batch_normalization();
    printf("\n");

    total++; passed += test_save_load();
    printf("\n");

    total++; passed += test_deterministic_forward_64();
    printf("\n");

    total++; passed += test_bptt_gradient_check_64();
    printf("\n");

    total++; passed += test_deterministic_forward_128();
    printf("\n");

    total++; passed += test_bptt_gradient_check_128();
    printf("\n");

    /* ===== 26-feature tests ===== */
    printf("--- 26-feature tests ---\n\n");
    total++; passed += test_forward_26_features();
    printf("\n");

    total++; passed += test_backward_26_features();
    printf("\n");

    total++; passed += test_deterministic_forward_26();
    printf("\n");

    total++; passed += test_bptt_gradient_check_26();
    printf("\n");

    /* ===== Data I/O tests ===== */
    printf("--- Data I/O tests ---\n\n");
    total++; passed += test_load_raw_basic();
    printf("\n");

    total++; passed += test_load_raw_chronological();
    printf("\n");

    total++; passed += test_aggregate_h1_identity();
    printf("\n");

    total++; passed += test_aggregate_h2();
    printf("\n");

    total++; passed += test_aggregate_h4();
    printf("\n");

    total++; passed += test_aggregate_h8();
    printf("\n");

    total++; passed += test_aggregate_double_h2();
    printf("\n");

    /* ===== Secondary-data (sdata) tests ===== */
    printf("--- Secondary-data (sdata) tests ---\n\n");
    total++; passed += test_intersect_raw_full_overlap();
    printf("\n");

    total++; passed += test_intersect_raw_partial();
    printf("\n");

    total++; passed += test_intersect_raw_empty();
    printf("\n");

    total++; passed += test_compute_raw_features_base_only();
    printf("\n");

    total++; passed += test_alloc_dataset_ext_null_secondary();
    printf("\n");

    total++; passed += test_alloc_dataset_ext_with_secondary();
    printf("\n");

    total++; passed += test_forward_39_features();
    printf("\n");

    total++; passed += test_backward_39_features();
    printf("\n");

    /* ===== OHLC-specific tests ===== */
    printf("--- OHLC tests ---\n\n");
    total++; passed += test_training_loop_ohlc();
    printf("\n");

    total++; passed += test_volume_clip_and_label_norm_ohlc();
    printf("\n");

    total++; passed += test_deterministic_ohlc();
    printf("\n");

    /* ===== Direction-specific tests ===== */
    printf("--- Direction tests ---\n\n");
    total++; passed += test_direction_labels();
    printf("\n");

    total++; passed += test_training_loop_direction();
    printf("\n");

    total++; passed += test_sigmoid_gradient_check();
    printf("\n");

    total++; passed += test_deterministic_direction();
    printf("\n");

    /* ===== sdata fix tests ===== */
    printf("--- sdata fix tests ---\n\n");
    total++; passed += test_mtf_features_finite();
    printf("\n");

    total++; passed += test_mtf_features_bounded();
    printf("\n");

    total++; passed += test_mtf_minmax_no_garbage();
    printf("\n");

    total++; passed += test_sdata_intersection_chronological();
    printf("\n");

    total++; passed += test_sdata_intersection_size();
    printf("\n");

    total++; passed += test_sdata_39_features_finite();
    printf("\n");

    total++; passed += test_sdata_iteration_count();
    printf("\n");

    /* ===== timeframe alignment tests ===== */
    printf("--- timeframe alignment tests ---\n\n");
    total++; passed += test_aggregate_h4_aligned();
    printf("\n");

    total++; passed += test_aggregate_h2_aligned();
    printf("\n");

    total++; passed += test_aggregate_h8_aligned();
    printf("\n");

    total++; passed += test_aggregate_h4_after_gap();
    printf("\n");

    total++; passed += test_aggregate_partial_group_kept();
    printf("\n");

    /* ===== datetime shift tests ===== */
    printf("--- datetime shift tests ---\n\n");
    total++; passed += test_datetime_shift_h1();
    printf("\n");

    total++; passed += test_datetime_shift_h4();
    printf("\n");

    total++; passed += test_datetime_shift_month_boundary();
    printf("\n");

    total++; passed += test_datetime_shift_h1_identity();
    printf("\n");

    /* ===== parameters file tests ===== */
    printf("--- parameters file tests ---\n\n");
    total++; passed += test_parameters_load();
    printf("\n");

    total++; passed += test_parameters_blank_defaults();
    printf("\n");

    total++; passed += test_parameters_cfg_loads();
    printf("\n");

    printf("========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
