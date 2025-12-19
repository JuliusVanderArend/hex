#include <vector>
//
// Created by Julius on 08/12/2025.
//
struct Node {
    int move_idx;
    int player_just_moved;
    Node* parent;
    std::vector<Node*> children;

    int visits = 0;
    double wins = 0.0;

    std::vector<int> unexpanded_moves;

    Node(int move, int player, Node* p)
        : move_idx(move), player_just_moved(player), parent(p) {}

    ~Node() { for (auto c : children) delete c; }

    bool isFullyExpanded() const { return unexpanded_moves.empty(); }
    bool isTerminal() const { return unexpanded_moves.empty() && children.empty(); }
};