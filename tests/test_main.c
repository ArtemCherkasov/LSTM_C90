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

    printf("========================================\n");
    printf("Results: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
