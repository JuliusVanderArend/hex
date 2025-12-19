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

// ChildEntry: statistics stored on the edge parent -> child
struct ChildEntry {
    int move = -1;
    Node* child = nullptr;
    float P = 0.0f;

    // Stats are now protected by the Parent Node's mutex
    int N = 0;
    float W = 0.0f;
    int virtualLoss = 0;    // [New] Tracks pending threads on this edge

    // Q-Value with Virtual Loss
    // We treat virtual losses as if they were visits with a losing value (-1.0)
    float Q(float virtualLossWeight = 1.0) const {
        int effectiveN = N + virtualLoss;
        if (effectiveN == 0) return 0.0f;

        // Subtract virtual loss from W (assuming W is relative perspective)
        // Effectively dragging the average down temporarily
        float effectiveW = W - (virtualLoss * virtualLossWeight);
        return effectiveW / effectiveN;
    }
};

struct Node {
    Node* parent = nullptr;
    std::vector<ChildEntry> children;
    bool expanded = false;
    int visits = 0; // Total visits through this node

    // [New] Mutex to protect 'children' stats and 'expanded' state during concurrent access
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
    double virtualLossWeight = 1.0; // Penalty strength for parallel paths

    MCTS() = default;
    ~MCTS() = default;

    // Main Search Entry Point
    SearchResult searchWithPolicy(const Position& rootPos, InferenceServer& server, int totalIterations) {
        Node* root = new Node(nullptr);

        // 1. Initial Root Expansion (Single-threaded)
        auto [rootPolicy, rootVal] = server.evaluate(rootPos);
        expandNode(root, rootPos, rootPolicy);

        // 2. Determine Thread Count
        // Use hardware concurrency, or match your Batch Size (e.g., 32)
        int numThreads = 1024;//std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 1;

        // Ensure at least 1 iteration per thread
        if (numThreads > totalIterations) numThreads = totalIterations;

        std::vector<std::thread> threads;
        std::atomic<int> iterationsCounter{0};

        // 3. Spawn Workers
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back([&]() {
                while (true) {
                    // Claim an iteration
                    int it = iterationsCounter.fetch_add(1);
                    if (it >= totalIterations) break;

                    worker_step(root, rootPos, server);
                }
            });
        }

        // 4. Wait for completion
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        // 5. Build Result
        SearchResult result = buildResult(root);
        delete root;
        return result;
    }

private:
    // The core loop for a single thread
    void worker_step(Node* root, Position pos, InferenceServer& server) {
        Node* node = root;
        std::vector<int> pathIndices;

        // --- 1. SELECTION ---
        while (true) {
            std::unique_lock<std::mutex> lock(node->mutex);

            if (!node->expanded || pos.getLegalMoves().empty()) {
                // Leaf reached
                break;
            }

            int idx = select_puct(node);

            if (idx < 0 || idx >= (int)node->children.size()) break;

            ChildEntry& entry = node->children[idx];

            // Apply Virtual Loss (Atomic via Lock)
            entry.virtualLoss++;
            node->visits++; // Increment parent visits speculatively

            pos.makeMove(entry.move);
            pathIndices.push_back(idx);

            // Lazy Child Creation
            if (!entry.child) {
                entry.child = new Node(node);
            }
            Node* nextNode = entry.child;

            lock.unlock(); // Release lock before descending
            node = nextNode;
        }

        // --- 2. EVALUATION & EXPANSION ---
        float value = 0.0f;
        int winner = pos.getWinner();

        if (winner != -1) {
            value = (winner == pos.sideToMove) ? 1.0f : -1.0f;
        } else {
            // Blocking Call to Server
            auto result = server.evaluate(pos);
            const std::vector<float>& policy = result.first;
            value = result.second;

            // Expand under lock
            std::lock_guard<std::mutex> lock(node->mutex);
            if (!node->expanded) {
                expandNode(node, pos, policy);
            }
        }

        // --- 3. BACKPROPAGATION ---
        backpropagate(root, pathIndices, value);
    }

    int select_puct(Node* node) {
        // NOTE: This function assumes the caller holds node->mutex
        int bestIdx = -1;
        double bestScore = -std::numeric_limits<double>::infinity();
        double sqrtVisits = std::sqrt((double)node->visits);
        double expl = cpuct * sqrtVisits;

        for (int i = 0; i < (int)node->children.size(); ++i) {
            const ChildEntry& e = node->children[i];

            // Use Q with Virtual Loss penalty
            double Q_val = e.Q(virtualLossWeight);

            // FPU: First Play Urgency
            if (e.N + e.virtualLoss == 0) {
                 Q_val = 0.0;
            }

            // Exploration Term (using effective N)
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
        // Caller must hold node->mutex
        std::vector<int> legal = pos.getLegalMoves();

        static constexpr int TOP_K = 24;
        static constexpr float MIN_P = 1e-6f;

        // 1. Rank legal moves by policy probability
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

        // Fallback: if all probabilities are tiny / zero
        if (ranked.empty()) {
            for (int mv : legal) {
                ranked.emplace_back(1.0f, mv);
            }
        }

        // 2. Keep only Top-K moves
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

        // 3. Normalize priors
        float sumP = 0.0f;
        for (int i = 0; i < K; ++i) sumP += ranked[i].first;
        if (sumP <= 0.0f) sumP = 1.0f;

        // 4. Create children
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
        // Root visits were incremented during Selection, so we don't do it here
        // or we need to be careful not to double count.
        // In this implementation, we incremented speculatively in Selection.

        for (int idx : pathIndices) {
            std::lock_guard<std::mutex> lock(node->mutex);

            ChildEntry& edge = node->children[idx];

            // Remove Virtual Loss
            edge.virtualLoss--;

            // Update Real Stats
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

        // No lock needed here as threads are joined
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