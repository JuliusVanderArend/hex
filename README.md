This repo contains the core C++ implementation of my Hex playing agent completed for COMP34111 (AI and Games) : a performance-oriented search engine paired with neural network inference via ONNX Runtime. The agent takes inspiration from AlphaGo and acheives a very high standard of play, reliably beating Mohex (pre RL state of the art). 
 
### Highlights
- Neural evaluation through ONNX Runtime for policy/value guidance (CPU by default; GPU-capable builds can be supported depending on your ONNX Runtime setup).
- Optimized Multithreaded MCTS with batched inference architecture maximises throughput of game tree search.
- Reusable core: the same engine used for interactive play, integration with Go Text Protocol and self-play/training utilities.
 

Build instructions (recommended)
The project is set up with CMake and expects ONNX Runtime headers and a shared library to be available in the Group49 directory structure (as configured in the provided CMakeLists.txt).
From the Group49 directory:
``` bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
This produces:
- (the main playing agent executable) `cpp_agent`
- `Group49SelfPlay` (self-play executable for automated games/data generation)

### Running
Typical usage depends on your tournament/driver wrapper, but the main deliverables are:
- **`cpp_agent`**: runs the agent as a standalone executable entry point.
- **`Group49SelfPlay`**: runs repeated games for evaluation and data generation.

If you’re integrating with an external arbiter, ensure the working directory contains the expected runtime assets (e.g., model files, ONNX Runtime `.so` present and discoverable via RPATH as configured).

### Troubleshooting
- **ONNX headers not found**: the build expects ONNX Runtime headers under `./include` (e.g., ). `onnxruntime/core/session/onnxruntime_cxx_api.h`
- **`libonnxruntime.so.1` not found**: the build expects the shared library in the **Group49** project root (as currently configured).
- **Runtime library loading issues**: the CMake configuration enables `$ORIGIN` RPATH so the executable can find shared libraries placed alongside it. If you move binaries, keep the `.so` files with them or adjust `LD_LIBRARY_PATH`.
