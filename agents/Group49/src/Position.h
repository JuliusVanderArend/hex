//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_POSITION_H
#define GROUP49_POSITION_H
#include <vector>

#include "../src/Util.h"

namespace engine {

    struct HexDSU {
        // 0-120: Board Cells
        // 121: Virtual START (Top for Red, Left for Blue)
        // 122: Virtual END   (Bottom for Red, Right for Blue)
        mutable uint8_t parent[128]; // 'mutable' allows path compression in const methods

        // Reset to initial state (all disconnected)
        void reset() {
            for (int i = 0; i < 128; ++i) parent[i] = i;
        }

        // Find with Path Compression
        Move find(uint8_t i) const {
            if (parent[i] == i) return i;
            return parent[i] = find(parent[i]);
        }

        // Unite two sets
        void unite(uint8_t i, uint8_t j) {
            uint8_t root_i = find(i);
            uint8_t root_j = find(j);
            if (root_i != root_j) {
                // Simple linking (Optimization: Union by rank could go here,
                // but for size 128, it's overkill and adds memory)
                parent[root_i] = root_j;
            }
        }

        // Check if connected
        bool isConnected(uint8_t start, uint8_t end) const {
            return find(start) == find(end);
        }
    };


    class Position {
    public:
        explicit Position(int sideToMove);

        void makeMove(Move move);
        void unmakeMove(Move move);
        std::vector<int> getLegalMoves() const;
        void makeRandomRolloutMove(FastRand& rng);
        Move getRandomLegalMove(FastRand& rng) const;
        int getWinner() const;


        void printBitboard(Board board) const;
        void printPosition() const;

        int sideToMove = 0;

        int moveCount = 0;
        HexDSU dsus[2];
    private:
        Board boards[2]; //frist board is us, second is them (always transposed)
        Board occupancy = 0;
        // Board occupancyTranspose = 0;



        bool isWon(Board* board);

        static inline void setHex(Move move, Board* board) {
            *board |= ((Board)1 << move);
        }
        static inline void unsetHex(Move move, Board* board) {
            *board &= ~((Board)1 << move);
        }
        static inline bool hasBit(Board board, Move index) {
            return (board >> index) & 1;
        }
    };
} // engine

#endif //GROUP49_POSITION_H