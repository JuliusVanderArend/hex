#ifndef GROUP49_MCTS_H
#define GROUP49_MCTS_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <iostream>
#include <utility>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include "Position.h"
#include "Util.h"
#include "InferenceServer.h"

namespace engine {

struct Node;

struct ChildEntry {
    int move = -1;
    Node* child = nullptr;
    float P = 0.0f;

    int N = 0;
    float W = 0.0f;
    int virtualLoss = 0;

    float Q(float virtualLossWeight = 1.0) const {
        int effectiveN = N + virtualLoss;
        if (effectiveN == 0) return 0.0f;

        float effectiveW = W - (virtualLoss * virtualLossWeight);
        return effectiveW / effectiveN;
    }
};

struct Node {
    Node* parent = nullptr;
    std::vector<ChildEntry> children;
    bool expanded = false;
    int visits = 0;

    std::mutex mutex;

    Node(Node* p = nullptr) : parent(p) {}

    ~Node() {
        for (auto &ce : children) {
            if (ce.child) delete ce.child;
            ce.child = nullptr;
        }
    }
};

struct SearchResult {
    int bestMove = -1;
    std::array<double, BOARD_AREA> policy{};
    float rootValue = 0.0f;
};

class MCTS {
public:
    double cpuct = 1.0;
    double virtualLossWeight = 1.0;

    MCTS() = default;
    ~MCTS() = default;

    SearchResult searchWithPolicy(const Position& rootPos, InferenceServer& server, int totalIterations) {
        Node* root = new Node(nullptr);

        auto [rootPolicy, rootVal] = server.evaluate(rootPos);
        expandNode(root, rootPos, rootPolicy);

        int numThreads = 192;//std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;

        if (numThreads > totalIterations) numThreads = totalIterations;

        std::vector<std::thread> threads;
        std::atomic<int> iterationsCounter{0};

        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back([&]() {
                while (true) {
                    int it = iterationsCounter.fetch_add(1);
                    if (it >= totalIterations) break;

                    worker_step(root, rootPos, server);
                }
            });
        }

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        SearchResult result = buildResult(root);
        delete root;
        return result;
    }

private:
    void worker_step(Node* root, Position pos, InferenceServer& server) {
        Node* node = root;
        std::vector<int> pathIndices;

        while (true) {
            std::unique_lock<std::mutex> lock(node->mutex);

            if (!node->expanded || pos.getLegalMoves().empty()) {
                break;
            }

            int idx = select_puct(node);

            if (idx < 0 || idx >= (int)node->children.size()) break;

            ChildEntry& entry = node->children[idx];

            entry.virtualLoss++;
            node->visits++;

            pos.makeMove(entry.move);
            pathIndices.push_back(idx);

            if (!entry.child) {
                entry.child = new Node(node);
            }
            Node* nextNode = entry.child;

            lock.unlock();
            node = nextNode;
        }

        float value = 0.0f;
        int winner = pos.getWinner();

        if (winner != -1) {
            value = (winner == (1 - pos.sideToMove)) ? 1.0f : -1.0f;
        } else {
            auto result = server.evaluate(pos);
            const std::vector<float>& policy = result.first;
            value = result.second;

            std::lock_guard<std::mutex> lock(node->mutex);
            if (!node->expanded) {
                expandNode(node, pos, policy);
            }
        }

        backpropagate(root, pathIndices, value);
    }

    int select_puct(Node* node) {
        int bestIdx = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        double sqrtVisits = std::sqrt((double)node->visits);
        double expl = cpuct * sqrtVisits;

        for (int i = 0; i < (int)node->children.size(); ++i) {
            const ChildEntry& e = node->children[i];

            double Q_val = e.Q(virtualLossWeight);

            if (e.N + e.virtualLoss == 0) {
                 Q_val = 0.0;
            }

            double denom = 1.0 + e.N + e.virtualLoss;
            double U_val = expl * e.P / denom;

            double score = Q_val + U_val;

            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    void expandNode(Node* node, const Position& pos, const std::vector<float>& policy_full) {
        std::vector<int> legal = pos.getLegalMoves();

        static constexpr int TOP_K = 40;
        static constexpr float MIN_P = 1e-6f;

        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(legal.size());

        for (int mv : legal) {
            float p = (mv >= 0 && mv < (int)policy_full.size())
                        ? policy_full[mv]
                        : 0.0f;

            if (p > MIN_P) {
                ranked.emplace_back(p, mv);
            }
        }

        if (ranked.empty()) {
            for (int mv : legal) {
                ranked.emplace_back(1.0f, mv);
            }
        }

        int K = std::min(TOP_K, (int)ranked.size());

        std::partial_sort(
            ranked.begin(),
            ranked.begin() + K,
            ranked.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            }
        );

        node->children.clear();
        node->children.reserve(K);

        float sumP = 0.0f;
        for (int i = 0; i < K; ++i) sumP += ranked[i].first;
        if (sumP <= 0.0f) sumP = 1.0f;

        for (int i = 0; i < K; ++i) {
            float p = ranked[i].first / sumP;
            int mv = ranked[i].second;

            node->children.push_back({
                mv,
                nullptr,
                p,
                0,      // N
                0.0f,   // W
                0       // virtualLoss
            });
        }

        node->expanded = true;
    }

    void backpropagate(Node* root, const std::vector<int>& pathIndices, float leafValue) {
        double valueForParent = -leafValue;

        Node* node = root;

        for (int idx : pathIndices) {
            std::lock_guard<std::mutex> lock(node->mutex);

            ChildEntry& edge = node->children[idx];

            edge.virtualLoss--;

            edge.N++;
            edge.W += valueForParent;

            node = edge.child;
            valueForParent = -valueForParent;
        }
    }

    SearchResult buildResult(Node* root) {
        SearchResult result;
        result.policy.fill(0.0);
        double totalN = 0.0;
        int bestMove = -1;
        int maxN = -1;

        for (const auto& ce : root->children) {
            result.policy[ce.move] = ce.N;
            totalN += ce.N;
            if (ce.N > maxN) {
                maxN = ce.N;
                bestMove = ce.move;
            }
        }
        if (totalN > 0) {
            for (double& p : result.policy) p /= totalN;

            double rootW = 0.0;
            for (const auto& ce : root->children) rootW += ce.W;
            result.rootValue = (float)(rootW / totalN);
        }
        result.bestMove = bestMove;
        return result;
    }
};

} // namespace engine

#endif // GROUP49_MCTS_H