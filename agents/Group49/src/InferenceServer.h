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
    struct InferenceRequest {
        Position pos;
        std::promise<std::pair<std::vector<float>, float>> promise;
        explicit InferenceRequest(const Position& p) : pos(p) {}
    };

    class InferenceServer {
        Inference& net;
        std::queue<InferenceRequest> queue;
        std::mutex queue_mutex;
        std::condition_variable cv_server;

        bool running = true;
        std::thread server_thread;

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

        std::pair<std::vector<float>, float> evaluate(const Position& pos) {
            std::future<std::pair<std::vector<float>, float>> future;

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue.emplace(pos);
                future = queue.back().promise.get_future();
            }

            cv_server.notify_one();

            return future.get();
        }


    private:
        float runSimulation(Position scratchPos) {
            FastRand rng;

            int myColor = scratchPos.sideToMove;

            while (scratchPos.getWinner() == -1) {
                scratchPos.makeRandomRolloutMove(rng);
            }
            return (scratchPos.getWinner() == myColor) ? 1.0f : -1.0f;
        }
        void loop() {
            while (true) {
                std::vector<InferenceRequest> batch;

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    cv_server.wait_for(lock, BATCH_TIMEOUT, [this] {
                        return queue.size() >= MAX_BATCH_SIZE || !running;
                    });

                    if (!running && queue.empty()) break;

                    while (!queue.empty() && batch.size() < MAX_BATCH_SIZE) {
                        batch.push_back(std::move(queue.front()));
                        queue.pop();
                    }
                }

                if (batch.empty()) continue;

                std::vector<Position> positions;
                positions.reserve(batch.size());
                for (const auto& req : batch) positions.push_back(req.pos);

                auto results = net.predictBatch(positions);

                for (size_t i = 0; i < batch.size(); ++i) {
                    batch[i].promise.set_value(results[i]);
                }
            }
        }
    };
}
#endif //GROUP49_INFERENCESERVER_H