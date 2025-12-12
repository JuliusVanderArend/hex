#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "Position.h"

class Inference {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info;

    // Fixed Input/Output Names
    const char* input_names[1] = {"state"};
    const char* output_names[2] = {"policy", "value"};


public:
    Inference(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_ERROR, "HexBot"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = Ort::Session(env, model_path.c_str(), session_options);

    }

    // --- EXISTING SINGLE PREDICT (Optional, kept for tests) ---
    std::pair<std::vector<float>, float> predict(const engine::Position& pos) {
        return predictBatch({pos})[0]; // Reuse the batch logic!
    }

    // --- NEW: BATCH PREDICTION ---
    // Used by InferenceServer to process multiple positions at once
    std::vector<std::pair<std::vector<float>, float>> predictBatch(const std::vector<engine::Position>& positions) {
        size_t batch_size = positions.size();
        if (batch_size == 0) return {};

        // 1. FLATTEN INPUTS
        // We need a contiguous vector of: [Pos1_Ch1..Ch6, Pos2_Ch1..Ch6, ...]
        std::vector<float> input_data;
        // Reserve memory: N * 6 * 11 * 11
        input_data.reserve(batch_size * 6 * 121);

        for (const auto& pos : positions) {
            std::vector<float> tensor = pos.toTensor(); // Returns 6*121 floats
            input_data.insert(input_data.end(), tensor.begin(), tensor.end());
        }

        // 2. CREATE TENSOR WITH DYNAMIC SHAPE
        // Shape: [BatchSize, 6, 11, 11]
        std::array<int64_t, 4> input_shape = { (int64_t)batch_size, 6, 11, 11 };

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(),
            input_data.size(),
            input_shape.data(),
            input_shape.size()
        );

        // 3. RUN INFERENCE
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );

        // 4. EXTRACT & SPLIT RESULTS
        float* policy_ptr = output_tensors[0].GetTensorMutableData<float>();
        float* value_ptr  = output_tensors[1].GetTensorMutableData<float>();

        std::vector<std::pair<std::vector<float>, float>> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            // A. Extract Policy Row (121 floats)
            // Pointer arithmetic: Start at i * 121
            std::vector<float> policy(policy_ptr + (i * 121), policy_ptr + ((i + 1) * 121));
            // C. Extract Value (1 float)
            float val = value_ptr[i];

            results.push_back({policy, val});
        }

        return results;
    }
};