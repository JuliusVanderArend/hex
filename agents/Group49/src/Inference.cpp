#include <onnxruntime_cxx_api.h>
#include <vector>
#include <array>
#include <iostream>
#include "Position.h"

// Adjust to match your model's expected input
// Example: Batch Size 1, 3 Channels (Red, Blue, Color), 11x11 Board
const int64_t BATCH_SIZE = 1;
const int64_t CHANNELS = 3; 
const int64_t HEIGHT = 11;
const int64_t WIDTH = 11;
const int64_t INPUT_SIZE = BATCH_SIZE * CHANNELS * HEIGHT * WIDTH;

class Inference {
    Ort::Env env;
    Ort::Session session{nullptr};
    
    // Memory Info for allocating tensors (CPU)
    Ort::MemoryInfo memory_info;

    // Fixed Input/Output Names (Must match your Python export!)
    const char* input_names[1] = {"input"};
    const char* output_names[2] = {"policy", "value"};

public:
    Inference(const std::wstring& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "HexBot"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) 
    {
        // 1. Configure Session Options (Graph Optimization)
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // 1 Thread per inference (we parallelize via batching)
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 2. Load Model
        // Note: On Windows use std::wstring for paths, Linux use std::string
        session = Ort::Session(env, model_path.c_str(), session_options);
    }

    // Returns {Policy Vector (121 floats), Value (float)}
    std::pair<std::vector<float>, float> predict(const engine::Position& pos) {
        
        // --- STEP 1: PREPARE INPUT ---
        // 1. Get the data directly from the position
        std::vector<float> input_tensor_values = pos.toTensor();

        // 2. Create the ONNX Tensor wrapper
        // Note: INPUT_SIZE must match input_tensor_values.size() (363)
        std::array<int64_t, 4> input_shape = {1, 3, 11, 11};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // --- STEP 2: RUN INFERENCE ---
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, 
            &input_tensor, 
            1, // Number of inputs
            output_names, 
            2  // Number of outputs
        );

        // --- STEP 3: EXTRACT OUTPUTS ---
        
        // Output 0: Policy (Logits or Probabilities) - Size 121
        float* policy_raw = output_tensors[0].GetTensorMutableData<float>();
        std::vector<float> policy(policy_raw, policy_raw + 121);

        // Output 1: Value (Win Probability) - Size 1
        float* value_raw = output_tensors[1].GetTensorMutableData<float>();
        float value = value_raw[0];

        return {policy, value};
    }
};