//
// Created by gamin on 12/12/2025.
//

#ifndef GROUP49_INFERENCESERVER_H
#define GROUP49_INFERENCESERVER_H
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <thread>
#include "Inference.cpp"
#include "Position.h"

class Inference;

namespace engine {
    // The "Ticket" a thread holds while waiting
    struct InferenceRequest {
        Position pos;
        std::promise<std::pair<std::vector<float>, float>> promise;
        explicit InferenceRequest(const Position& p) : pos(p) {}
    };

    class InferenceServer {
        Inference& net;

        // Queue State
        std::queue<InferenceRequest> queue;
        std::mutex queue_mutex;
        std::condition_variable cv_server; // Wakes up the server

        bool running = true;
        std::thread server_thread;

        // Tuning Parameters
        const size_t MAX_BATCH_SIZE = 128;
        const std::chrono::microseconds BATCH_TIMEOUT = std::chrono::microseconds(100); // 0.1ms

    public:
        InferenceServer(Inference& network) : net(network) {
            server_thread = std::thread(&InferenceServer::loop, this);
        }

        ~InferenceServer() {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                running = false;
            }
            cv_server.notify_one();
            if (server_thread.joinable()) server_thread.join();
        }

        // Called by MCTS Worker Threads
        // This function BLOCKS until the result is ready.
        std::pair<std::vector<float>, float> evaluate(const Position& pos) {
            std::future<std::pair<std::vector<float>, float>> future;

            {
                std::lock_guard<std::mutex> lock(queue_mutex);

                // push_back is tricky with promises (non-copyable), so we construct in-place
                queue.emplace(pos);
                future = queue.back().promise.get_future();
            }

            // Notify server we added something
            cv_server.notify_one();

            // BLOCK HERE
            return future.get();
        }
        // std::pair<std::vector<float>, float> evaluate(Position pos) {
        //     // 1. SIMULATION (Value)
        //     // Run a random rollout to determine the winner
        //     // We pass 'pos' by value so we can destroy this copy during simulation
        //     float value = runSimulation(pos);
        //
        //     // 2. UNIFORM POLICY
        //     // Classic MCTS has no prior knowledge, so P(s,a) is uniform.
        //     // We create a vector where valid moves are 1.0 (expandNode handles normalization).
        //     std::vector<float> uniformPolicy(BOARD_AREA, 0.0f);
        //
        //     // Optimization: We don't strictly need to fill this if expandNode
        //     // detects a zero-sum and falls back to uniform, but let's be explicit.
        //     for (int move : pos.getLegalMoves()) {
        //         uniformPolicy[move] = 1.0f;
        //     }
        //
        //     return {uniformPolicy, value};
        // }

    private:
        float runSimulation(Position scratchPos) {
            // Use thread-local RNG (FastRand from Util.h)
            // Crucial: Re-instantiating FastRand per call ensures unique seeds per thread
            FastRand rng;

            int myColor = scratchPos.sideToMove;

            while (scratchPos.getWinner() == -1) {
                // Check consistency/legality if debugging, otherwise optimize for speed
                // Use your Position's fast random move method
                // If you don't have makeRandomRolloutMove, use:
                // int m = scratchPos.getRandomLegalMove(rng);
                // scratchPos.makeMove(m);
                scratchPos.makeRandomRolloutMove(rng);
            }

            // Returns 1.0 if the player-to-move won, -1.0 if they lost.
            // (Standard Zero-Sum perspective)
            return (scratchPos.getWinner() == myColor) ? 1.0f : -1.0f;
        }
        void loop() {
            while (true) {
                std::vector<InferenceRequest> batch;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);

                    // Wait until:
                    // 1. We have a FULL batch
                    // 2. OR the timeout expires (latency protection)
                    // 3. OR we are shutting down
                    cv_server.wait_for(lock, BATCH_TIMEOUT, [this] {
                        return queue.size() >= MAX_BATCH_SIZE || !running;
                    });

                    if (!running && queue.empty()) break;

                    // Move items from Queue -> Local Batch
                    // We perform the inference outside the lock so workers can keep queuing
                    while (!queue.empty() && batch.size() < MAX_BATCH_SIZE) {
                        // Moving is efficient (transfer ownership of promise)
                        batch.push_back(std::move(queue.front()));
                        queue.pop();
                    }
                }

                if (batch.empty()) continue;

                // LOG
                static std::atomic<int> printed{0};
                if (printed++ < 20) {
                    std::cout << "[InferenceServer] batch size = "
                              << batch.size() << std::endl;
                }

                // --- BATCH INFERENCE ---
                std::vector<Position> positions;
                positions.reserve(batch.size());
                for (const auto& req : batch) positions.push_back(req.pos);

                auto results = net.predictBatch(positions);

                // --- WAKE UP WORKERS ---
                for (size_t i = 0; i < batch.size(); ++i) {
                    batch[i].promise.set_value(results[i]);
                }
            }
        }
    };
}
#endif //GROUP49_INFERENCESERVER_H