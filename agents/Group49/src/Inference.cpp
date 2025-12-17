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

    // Persistent buffers
    std::vector<float> input_buffer;
    std::array<int64_t, 4> input_shape;
    Ort::Value input_tensor{nullptr};

    const size_t MAX_BATCH = 128; // has to be the same as in InferenceServer

    // Fixed Input/Output Names
    const char* input_names[1] = {"state"};
    const char* output_names[2] = {"policy", "value"};


public:
    Inference(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_ERROR, "HexBot"),
      memory_info(Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
#ifdef USE_CUDA
        OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif
        session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = Ort::Session(env, model_path.c_str(), session_options);

        // Allocate persistent input buffer
        input_buffer.resize(MAX_BATCH * 6 * 121);

        // Default shape (batch size will change dynamically)
        input_shape = {1, 6, 11, 11};

        // Create persistent tensor
        input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_buffer.data(),
            input_buffer.size(),
            input_shape.data(),
            input_shape.size()
        );
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
        if (batch_size > MAX_BATCH) {
            std::cerr << "Batch too large: " << batch_size << std::endl;
            std::terminate();
        }


        // 1. Update dynamic batch size
        input_shape[0] = static_cast<int64_t>(batch_size);

        // 2. Fill persistent input buffer
        for (size_t i = 0; i < batch_size; ++i) {
            positions[i].toTensor(
                input_buffer.data() + i * 6 * 121
            );
        }

        // 3. Run inference (reuse tensor!)
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            2
        );

        // 4. Extract outputs
        float* policy_ptr = output_tensors[0]
            .GetTensorMutableData<float>();
        float* value_ptr = output_tensors[1]
            .GetTensorMutableData<float>();

        std::vector<std::pair<std::vector<float>, float>> results;
        results.reserve(batch_size);

        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<float> policy(
                policy_ptr + i * 121,
                policy_ptr + (i + 1) * 121
            );
            results.emplace_back(policy, value_ptr[i]);
        }

        return results;
    }
};