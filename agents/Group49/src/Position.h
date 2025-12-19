//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_POSITION_H
#define GROUP49_POSITION_H
#include <iostream>
#include <vector>

#include "../src/Util.h"

namespace engine {

    struct HexDSU {
        mutable uint8_t parent[128];

        void reset() {
            for (int i = 0; i < 128; ++i) parent[i] = i;
        }

        Move find(uint8_t i) const {
            if (parent[i] == i) return i;
            return parent[i] = find(parent[i]);
        }

        void unite(uint8_t i, uint8_t j) {
            uint8_t root_i = find(i);
            uint8_t root_j = find(j);
            if (root_i != root_j) {
                parent[root_i] = root_j;
            }
        }

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

        std::vector<float> toTensor() const;
        void toTensor(float* dst) const;

        void printBitboard(Board board) const;
        void printPosition() const;

        bool hasStone(int player, Move move) const;

        Board getBoardBits(int player) const { return boards[player]; }
        Board getOccupancyBits() const { return occupancy; }

        void loadFromSnapshot(const std::vector<std::string>& rows);

        int sideToMove = 0;

        int moveCount = 0;
        HexDSU dsus[2];
        bool isMoveLegal(int move) const {
            if (move < 0 || move >= BOARD_AREA) return false;
            return !((occupancy >> move) & 1);
        }

        void checkConsistency() const {
            for (int i = 0; i < BOARD_AREA; ++i) {
                bool redClaim = (boards[0] >> i) & 1;

                int r = i / BOARD_SIZE;
                int c = i % BOARD_SIZE;
                int transposedIdx = c * BOARD_SIZE + r;

                bool blueClaim = (boards[1] >> transposedIdx) & 1;

                if (redClaim && blueClaim) {
                    std::cerr << "\n========================================" << std::endl;
                    std::cerr << "CRITICAL ERROR: BOARD CORRUPTION DETECTED" << std::endl;
                    std::cerr << "========================================" << std::endl;
                    std::cerr << "Overlap at Index: " << i << " (Row " << r << ", Col " << c << ")" << std::endl;
                    std::cerr << "Red claims vertical index:   " << i << std::endl;
                    std::cerr << "Blue claims transposed index: " << transposedIdx << std::endl;
                    std::cerr << "Aborting program to prevent undefined behavior." << std::endl;
                    exit(1);
                }
            }
        }
    private:
        Board boards[2];
        Board occupancy = 0;
        Move lastMove = -1;



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
