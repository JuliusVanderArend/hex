#ifndef GROUP49_MCTS_H
#define GROUP49_MCTS_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <iostream>
#include <utility>

#include "Position.h"
#include "Util.h"

namespace engine {

struct Node; // Forward declaration


// ChildEntry: statistics stored on the edge parent → child
struct ChildEntry {
    int move = -1;          // Move index (0..BOARD_AREA-1)
    Node* child = nullptr;  // Created lazily during selection/expansion
    double P = 0.0;         // Prior probability from policy network (or uniform)
    int N = 0;              // Visit count
    double W = 0.0;         // Total accumulated value
    double Q() const { return N ? (W / N) : 0.0; }
};


// Node: represents a position reached by move_from_parent
// Edge statistics live in ChildEntry, not in Node itself.
struct Node {
    int move_from_parent = -1;    // Move that created this node (root = -1)
    int player_just_moved = -1;   // Player who played move_from_parent
    Node* parent = nullptr;

    std::vector<ChildEntry> children; // One entry per legal move after expansion
    bool expanded = false;            // Becomes true after expandNode()
    int visits = 0;                   // Number of visits to this node

    Node(int mv = -1, int player = -1, Node* p = nullptr)
        : move_from_parent(mv), player_just_moved(player), parent(p) {}

    ~Node() {
        for (auto &ce : children) {
            if (ce.child) delete ce.child;
            ce.child = nullptr;
        }
    }

    bool isFullyExpanded() const {
        if (!expanded) return false;
        for (const auto &ce : children) {
            if (ce.move >= 0 && ce.child == nullptr)
                return false;
        }
        return true;
    }
};


// SearchResult: returned by searchWithPolicy()
// policy[i] = normalized visit count for move i
struct SearchResult {
    int bestMove = -1;
    std::array<double, BOARD_AREA> policy{};
};


// MCTS CLASS
class MCTS {
public:
    double cpuct = 1.0;   // PUCT exploration constant

    FastRand mcts_rng;

    MCTS() = default;
    ~MCTS() = default;

    // Returns a move chosen by most visits
    int search(const Position& rootPos, int iterations) {
        return searchWithPolicy(rootPos, iterations).bestMove;
    }

    // Runs MCTS and returns move + visit distribution
    SearchResult searchWithPolicy(const Position& rootPos, int iterations) {
        Node* root = new Node(-1, rootPos.sideToMove ^ 1, nullptr);

        for (int it = 0; it < iterations; ++it) {
            Position pos = rootPos;

            // -------------------
            // Selection
            // -------------------
            std::vector<std::pair<Node*, int>> path;
            Node* node = root;

            while (node->expanded && !isTerminalPosition(pos)) {
                int idx = select_puct(node);
                if (idx < 0 || idx >= (int)node->children.size()) break;

                int mv = node->children[idx].move;
                pos.makeMove(mv);

                if (!node->children[idx].child) {
                    node->children[idx].child =
                        new Node(mv, pos.sideToMove ^ 1, node);
                }

                path.emplace_back(node, idx);
                node = node->children[idx].child;
            }

            // -------------------
            // Expansion + Evaluation
            // -------------------
            int winner = -1;
            if (isTerminalPosition(pos)) {
                winner = pos.getWinner();
            } else {
                // TODO: Replace uniform priors + rollout with CNN-based priors + value.
                std::vector<double> priors; // empty → uniform priors
                expandNode(node, pos, priors);

                // Rollout value (placeholder until CNN integration)
                winner = simulate(pos);
            }

            // -------------------
            // Backpropagation
            // -------------------
            backpropagate(path, node, winner);
        }

        // -------------------
        // Build final policy
        // -------------------
        SearchResult result;
        result.policy.fill(0.0);

        if (!root->expanded) {
            Position tmp = rootPos;
            expandNode(root, tmp, std::vector<double>{}); // uniform
        }

        double totalN = 0.0;
        int bestMove = -1;
        int bestN = -1;

        for (const auto &ce : root->children) {
            result.policy[ce.move] = ce.N;
            totalN += ce.N;
            if (ce.N > bestN) {
                bestN = ce.N;
                bestMove = ce.move;
            }
        }

        if (totalN > 0) {
            for (double &v : result.policy) v /= totalN;
        }

        result.bestMove = bestMove;
        delete root;
        return result;
    }

private:
    // Terminal detection using Position API
    bool isTerminalPosition(const Position& pos) {
        return pos.getWinner() != -1;
    }


    // Selection: choose child index by PUCT formula
    int select_puct(Node* node) {
        if (!node->expanded || node->children.empty()) return -1;

        double sumN = 0.0;
        for (const auto &ce : node->children) sumN += ce.N;
        double sqrt_sum = std::sqrt(std::max(1.0, sumN));

        int bestIdx = -1;
        double bestVal = -std::numeric_limits<double>::infinity();

        for (int i = 0; i < (int)node->children.size(); ++i) {
            const ChildEntry &e = node->children[i];
            double q = e.N ? (e.W / e.N) : 0.0;
            double u = cpuct * e.P * (sqrt_sum / (1 + e.N));
            double val = q + u;
            if (val > bestVal) {
                bestVal = val;
                bestIdx = i;
            }
        }
        return bestIdx;
    }


    // Expansion: populate children with legal moves and priors.
    // If priors is empty, uniform is used.
    void expandNode(Node* node, const Position& pos,
                    const std::vector<double>& priors)
    {
        std::vector<int> legal = pos.getLegalMoves();

        node->children.clear();
        node->children.reserve(legal.size());

        double sumP = 0.0;

        for (int mv : legal) {
            double p = 1.0;
            if (!priors.empty() && (size_t)mv < priors.size()) p = priors[mv];

            node->children.push_back(ChildEntry{mv, nullptr, p, 0, 0.0});
            sumP += p;
        }

        if (sumP <= 0.0) {
            double u = 1.0 / std::max<size_t>(1, legal.size());
            for (auto &c : node->children) c.P = u;
        } else {
            for (auto &c : node->children) c.P /= sumP;
        }

        node->expanded = true;
    }


    // Rollout simulation (placeholder until CNN value head)
    int simulate(Position& pos) {
        int immediate = pos.getWinner();
        if (immediate != -1) return immediate;

        while (true) {
            pos.makeRandomRolloutMove(mcts_rng);

            int just_moved = pos.sideToMove ^ 1;
            if (pos.dsus[just_moved].isConnected(V_START, V_END)) {
                return just_moved;
            }
            if (pos.moveCount >= BOARD_AREA) return 2; // draw (rare in Hex)
        }
    }


    // Backpropagation: update edge statistics (N, W) along path
    void backpropagate(const std::vector<std::pair<Node*, int>>& path,
                       Node* leafNode,
                       int winner)
    {
        // Update edges from leaf to root
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            Node* parent = it->first;
            int idx = it->second;

            if (idx < 0 || idx >= (int)parent->children.size()) continue;

            ChildEntry &edge = parent->children[idx];
            edge.N += 1;

            int player_who_moved =
                edge.child ? edge.child->player_just_moved
                           : (parent->player_just_moved ^ 1);

            if (winner == player_who_moved) edge.W += 1.0;
            else if (winner == 2) edge.W += 0.5;

            parent->visits += 1;
        }

        Node* cur = leafNode;
        while (cur) {
            cur->visits += 1;
            cur = cur->parent;
        }
    }
};

} // namespace engine

#endif // GROUP49_MCTS_H
