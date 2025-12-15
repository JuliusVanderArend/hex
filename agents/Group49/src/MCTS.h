//
// Created by Julius on 09/12/2025.
//

#ifndef GROUP49_MCTS_H
#define GROUP49_MCTS_H

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <array>
#include <iostream>
#include "../src/Position.h"
namespace engine {
    // Lightweight Tree Node
    // Memory footprint optimized: We do not store the full Board state.
    struct Node {
        int move_idx;          // The move that created this node
        int player_just_moved; // 0 (Red) or 1 (Blue)
        Node* parent;
        std::vector<Node*> children;

        // Statistics
        int visits = 0;
        double score = 0.0;    // Wins accumulated for the parent of this node

        // Expansion Management
        std::vector<int> unexpanded_moves;

        // Constructor: Populates legal moves immediately using bitscan
        Node(int move, int player, Node* parent_node, const Position& pos)
            : move_idx(move), player_just_moved(player), parent(parent_node) {

            // Use the fast bitscan helper from Position
            unexpanded_moves = pos.getLegalMoves();
        }

        ~Node() {
            for (Node* child : children) delete child;
        }

        bool isFullyExpanded() const {
            return unexpanded_moves.empty();
        }

        bool isTerminal() const {
            // Terminal if no moves left AND no children created
            return unexpanded_moves.empty() && children.empty();
        }
    };

struct SearchResult {
    int bestMove = -1;
    std::array<double, BOARD_AREA> policy{};
};

class MCTS {
public:
    // Parameters
    // C_PARAM: Exploration constant. sqrt(2) is theoretical,
    // but often 0.6-1.0 works better for games with branching factor ~100.
    double C_PARAM = 1.0;

    // Helper RNG for expansion choices (Position uses its own for rollouts)
    FastRand mcts_rng;

    // ------------------------------------------------------------------------
    // MAIN SEARCH FUNCTION
    // ------------------------------------------------------------------------
    int search(const Position& rootPos, int iterations) {
        return searchWithPolicy(rootPos, iterations).bestMove;
    }

    SearchResult searchWithPolicy(const Position& rootPos, int iterations) {
        // 1. Root Node Creation
        // The root represents the state *before* we make a move.
        // So 'player_just_moved' is the opponent of sideToMove.
        Node* root = new Node(-1, rootPos.sideToMove ^ 1, nullptr, rootPos);

        for (int i = 0; i < iterations; ++i) {
            // A. Clone the board (Trivial Copy: very fast)
            Position scratchPos = rootPos;

            // B. Selection
            Node* leaf = select(root, scratchPos);

            // C. Expansion

            // If the game isn't effectively over at this leaf, expand it.
            // Note: getWinner returns -1 if ongoing.
            if (scratchPos.getWinner() == -1 && !leaf->isTerminal()) {
                leaf = expand(leaf, scratchPos);
            }

            // D. Simulation
            // Run the optimized random rollout
            int winner = simulate(scratchPos);

            // E. Backpropagation
            backpropagate(leaf, winner);
        }

        // 4. Select Robust Child (Max Visits) and capture visit distribution
        SearchResult result;
        result.policy.fill(0.0);
        double totalVisits = 0.0;
        int maxVisits = -1;

        for (Node* child : root->children) {
            result.policy[child->move_idx] = static_cast<double>(child->visits);
            totalVisits += child->visits;

            if (child->visits > maxVisits) {
                maxVisits = child->visits;
                result.bestMove = child->move_idx;
            }
        }

        if (totalVisits > 0.0) {
            for (double& value : result.policy) {
                value /= totalVisits;
            }
        }

        delete root; // Clean up the entire tree
        return result;
    }

private:
    // --- SELECTION ---
    // Drills down the tree using UCT until it hits a node that is not fully expanded.
    // Side Effect: Updates 'pos' to match the state at the leaf.
    Node* select(Node* node, Position& pos) {
        while (node->isFullyExpanded() && !node->isTerminal()) {
            node = getBestUCTChild(node);
            pos.makeMove(node->move_idx);
        }
        return node;
    }

    // --- EXPANSION ---
    // Adds ONE new child to the tree.
    Node* expand(Node* node, Position& pos) {
        if (node->unexpanded_moves.empty()) return node; // Should not happen given logic above

        // 1. Pick a random move from the unexpanded list
        // This avoids bias in the expansion order
        int idx = mcts_rng.range(node->unexpanded_moves.size());
        int move = node->unexpanded_moves[idx];

        // 2. Efficiently remove from list (Swap with back + pop)
        node->unexpanded_moves[idx] = node->unexpanded_moves.back();
        node->unexpanded_moves.pop_back();

        // 3. Update the scratch board
        pos.makeMove(move);

        // 4. Create the new node
        // pos.sideToMove ^ 1 is the player who just made this move
        Node* newChild = new Node(move, pos.sideToMove ^ 1, node, pos);
        node->children.push_back(newChild);

        return newChild;
    }

    // --- SIMULATION ---
    // Pure random rollout using your optimized PDEP/DSU code
    int simulate(Position& pos) {
        // If the game ended during selection/expansion (e.g. instant win), return immediately
        int initial_winner = pos.getWinner();
        if (initial_winner != -1) return initial_winner;

        // Run until game over
        // We assume your makeRandomRolloutMove handles move limit/draws internally if needed
        while (true) {
            // Use the position's internal RNG or pass one in if needed
            // pos.makeRandomRolloutMove expects FastRand&.
            // We reuse mcts_rng for simplicity, or create a local one.
            // Using mcts_rng is fine since MCTS is single-threaded here.
            pos.makeRandomRolloutMove(mcts_rng);

            // Check ONLY the active player for efficiency (Optimization #2)
            // makeMove flipped the turn, so we check the player who just moved
            int just_moved = pos.sideToMove ^ 1;

            // Check DSU for that player
            if (pos.dsus[just_moved].isConnected(V_START, V_END)) {
                return just_moved;
            }

            // Safety check for full board (Draw)
            if (pos.moveCount >= BOARD_AREA) return 2;
        }
    }

    // --- BACKPROPAGATION ---
    void backpropagate(Node* node, int winner) {
        while (node != nullptr) {
            node->visits++;

            // Standard MCTS Scoring:
            // If the winner matches the player who made the move at this node,
            // it's a win for this node.
            if (winner == node->player_just_moved) {
                node->score += 1.0;
            } else if (winner == 2) {
                // Draw (rare/impossible in Hex)
                node->score += 0.5;
            }
            // Loss adds 0.0

            node = node->parent;
        }
    }

    // --- UCT HELPER ---
    Node* getBestUCTChild(Node* node) {
        Node* bestChild = nullptr;
        double bestValue = -std::numeric_limits<double>::max();

        // Pre-calculate log(N)
        double logParent = std::log(node->visits);

        for (Node* child : node->children) {
            // UCT Formula: (Wins / Visits) + C * sqrt(log(ParentVisits) / Visits)
            double exploitation = child->score / child->visits;
            double exploration  = C_PARAM * std::sqrt(logParent / child->visits);

            double uct = exploitation + exploration;

            if (uct > bestValue) {
                bestValue = uct;
                bestChild = child;
            }
        }
        return bestChild;
    }
    };
}
#endif //GROUP49_MCTS_H
