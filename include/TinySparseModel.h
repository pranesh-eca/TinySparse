#ifndef TINYSPARSE_MODEL_H
#define TINYSPARSE_MODEL_H

#include "TinySparse.h"
#include <limits.h>

namespace TinySparse {

enum class Storage : uint8_t { Dense, Csr };

// Hidden layers apply ReLU and a right shift. The final layer emits INT32 logits.
// CSR uses 8-bit columns and 16-bit row offsets, so input width <= 256 and
// nonzero count <= 65535. All pointed-to arrays must remain alive during use.
struct Layer {
    Storage storage;
    uint16_t input_size;
    uint16_t output_size;
    const int8_t* weights;      // Dense weights or CSR nonzero values
    const uint8_t* columns;     // CSR only
    const uint16_t* offsets;    // CSR only; output_size + 1 elements
    const int32_t* bias;
    uint16_t nonzeros;          // CSR only
    uint8_t shift;              // Hidden layers only; final layer uses 0
};

struct Model {
    const Layer* layers;
    uint16_t layer_count;
};

enum class ModelStatus : uint8_t { Ok, InvalidModel, InsufficientWorkspace };

inline uint16_t max_hidden_width(const Model& model) {
    uint16_t largest = 0;
    if (!model.layers || model.layer_count == 0) return 0;
    for (uint16_t i = 0; i + 1 < model.layer_count; ++i) {
        if (model.layers[i].output_size > largest) largest = model.layers[i].output_size;
    }
    return largest;
}

// Two alternating INT8 buffers allow any number of hidden layers.
inline size_t required_workspace_bytes(const Model& model) {
    return static_cast<size_t>(max_hidden_width(model)) * 2;
}

// Call once after loading/generated model data, before repeated inference.
// A conservative bound rejects any row that could overflow INT32 for an INT8 input.
inline bool validate_model(const Model& model) {
    if (!model.layers || model.layer_count == 0) return false;
    for (uint16_t i = 0; i < model.layer_count; ++i) {
        const Layer& layer = model.layers[i];
        if (!layer.input_size || !layer.output_size || !layer.weights || !layer.bias) return false;
        if (i && model.layers[i - 1].output_size != layer.input_size) return false;
        if (i + 1 == model.layer_count) {
            if (layer.shift != 0) return false;
        } else if (layer.shift > 30) {
            return false;
        }
        if (layer.storage != Storage::Dense && layer.storage != Storage::Csr) return false;
        if (layer.storage == Storage::Csr) {
            if (layer.input_size > 256 || !layer.columns || !layer.offsets || layer.offsets[0] != 0 ||
                layer.offsets[layer.output_size] != layer.nonzeros) return false;
        }
        for (uint16_t row = 0; row < layer.output_size; ++row) {
            int64_t bound = layer.bias[row] < 0 ? -static_cast<int64_t>(layer.bias[row]) : layer.bias[row];
            if (layer.storage == Storage::Dense) {
                for (uint16_t col = 0; col < layer.input_size; ++col) {
                    int32_t weight = layer.weights[static_cast<size_t>(row) * layer.input_size + col];
                    bound += 128LL * (weight < 0 ? -weight : weight);
                }
            } else {
                const uint16_t start = layer.offsets[row];
                const uint16_t end = layer.offsets[row + 1];
                if (end < start || end > layer.nonzeros) return false;
                for (uint16_t k = start; k < end; ++k) {
                    if (layer.columns[k] >= layer.input_size) return false;
                    int32_t weight = layer.weights[k];
                    bound += 128LL * (weight < 0 ? -weight : weight);
                }
            }
            if (bound > INT32_MAX) return false;
        }
    }
    return true;
}

// Requires a model that passed validate_model(). Buffers must not overlap.
// workspace may be null only when there are no hidden layers.
inline ModelStatus run_model(const Model& model, const int8_t* input,
                             int8_t* workspace, size_t workspace_bytes,
                             int32_t* logits) {
    if (!model.layers || model.layer_count == 0 || !input || !logits) return ModelStatus::InvalidModel;
    const uint16_t width = max_hidden_width(model);
    if (workspace_bytes < static_cast<size_t>(width) * 2 || (width && !workspace)) {
        return ModelStatus::InsufficientWorkspace;
    }
    const int8_t* source = input;
    for (uint16_t i = 0; i < model.layer_count; ++i) {
        const Layer& layer = model.layers[i];
        const bool final_layer = i + 1 == model.layer_count;
        if (final_layer) {
            if (layer.storage == Storage::Dense) {
                dense_fc_logits(source, layer.weights, layer.bias, logits, layer.input_size, layer.output_size);
            } else {
                sparse_fc_logits(source, layer.weights, layer.columns, layer.offsets, layer.bias, logits, layer.output_size);
            }
        } else {
            int8_t* target = workspace + (i % 2) * width;
            if (layer.storage == Storage::Dense) {
                dense_fc_relu(source, layer.weights, layer.bias, target, layer.input_size, layer.output_size, layer.shift);
            } else {
                sparse_fc_relu(source, layer.weights, layer.columns, layer.offsets, layer.bias, target, layer.output_size, layer.shift);
            }
            source = target;
        }
    }
    return ModelStatus::Ok;
}

} // namespace TinySparse

#endif // TINYSPARSE_MODEL_H
