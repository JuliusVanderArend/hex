//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_POSITION_H
#define GROUP49_POSITION_H
#include "../src/Util.h"

namespace engine {
    class Position {
    public:
        explicit Position(int sideToMove);

        void makeMove(Move move);
        void unmakeMove(Move move);
        void makeRandomRolloutMove(FastRand& rng);
        int getRandomLegalMove(FastRand& rng);
        int getWinner();


        void printBitboard(Board board) const;
        void printPosition() const;

        int sideToMove = 0;

        int moveCount = 0;
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
        // static inline Move transposeMove(Move move) {
        //     uint_fast8_t x = move % BOARD_SIZE;
        //     uint_fast8_t y = move / BOARD_SIZE;
        //     Move transposed = y*BOARD_SIZE + x;
        //     return transposed;
        // }
    };
} // engine

#endif //GROUP49_POSITION_H