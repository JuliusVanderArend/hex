#include <vector>
//
// Created by Julius on 08/12/2025.
//
struct Node {
    int move_idx;          // The move that led to this state
    int player_just_moved; // 0 (Red) or 1 (Blue)
    Node* parent;
    std::vector<Node*> children;

    int visits = 0;
    double wins = 0.0;

    // Tracks which moves are possible from this state but not yet expanded
    std::vector<int> unexpanded_moves;

    Node(int move, int player, Node* p)
        : move_idx(move), player_just_moved(player), parent(p) {}

    ~Node() { for (auto c : children) delete c; }

    bool isFullyExpanded() const { return unexpanded_moves.empty(); }
    bool isTerminal() const { return unexpanded_moves.empty() && children.empty(); }
};