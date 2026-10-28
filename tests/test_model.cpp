#include <assert.h>
#include <stdint.h>
#include "TinySparseModel.h"
#include "model_weights.h"

int main() {
    using namespace TinySparse;
    const Layer cwru_layers[] = {
        {Storage::Csr, 128, FC1_OUT_DIM, fc1_values, fc1_col_indices, fc1_row_offsets,
         fc1_bias, static_cast<uint16_t>(sizeof(fc1_values)), FC1_SHIFT},
        {Storage::Csr, FC1_OUT_DIM, FC2_OUT_DIM, fc2_values, fc2_col_indices, fc2_row_offsets,
         fc2_bias, static_cast<uint16_t>(sizeof(fc2_values)), FC2_SHIFT},
        {Storage::Dense, FC3_IN_DIM, FC3_OUT_DIM, fc3_weights_dense, nullptr, nullptr,
         fc3_bias, 0, 0}
    };
    const Model cwru{cwru_layers, 3};
    assert(validate_model(cwru));
    assert(required_workspace_bytes(cwru) == 64);
    int8_t work[64] = {};
    int32_t logits[FC3_OUT_DIM] = {};
    assert(run_model(cwru, golden_input, work, sizeof(work), logits) == ModelStatus::Ok);
    for (uint16_t i = 0; i < FC3_OUT_DIM; ++i) assert(logits[i] == golden_logits[i]);
    assert(run_model(cwru, golden_input, work, sizeof(work) - 1, logits) == ModelStatus::InsufficientWorkspace);

    const int8_t hidden_weights[] = {1, 0, 0, 0, 1, -1};
    const int32_t hidden_bias[] = {0, 0};
    const int8_t final_values[] = {3, -2};
    const uint8_t final_columns[] = {0, 0};
    const uint16_t final_offsets[] = {0, 1, 2};
    const int32_t final_bias[] = {1, 4};
    const Layer mixed_layers[] = {
        {Storage::Dense, 3, 2, hidden_weights, nullptr, nullptr, hidden_bias, 0, 0},
        {Storage::Csr, 2, 2, final_values, final_columns, final_offsets, final_bias, 2, 0}
    };
    const Model mixed{mixed_layers, 2};
    const int8_t input[] = {2, 3, 4};
    int8_t mixed_work[4] = {};
    int32_t mixed_logits[2] = {};
    assert(validate_model(mixed));
    assert(run_model(mixed, input, mixed_work, sizeof(mixed_work), mixed_logits) == ModelStatus::Ok);
    assert(mixed_logits[0] == 7 && mixed_logits[1] == 0);

    const uint8_t bad_column[] = {2, 0};
    const Layer bad_layers[] = {
        mixed_layers[0],
        {Storage::Csr, 2, 2, final_values, bad_column, final_offsets, final_bias, 2, 0}
    };
    assert(!validate_model({bad_layers, 2}));

    int32_t many_logits[300] = {};
    many_logits[280] = 7;
    assert(argmax(many_logits, 300) == 280);
    return 0;
}
