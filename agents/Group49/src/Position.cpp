//
// Created by Julius on 28/11/2025.
//
#include <immintrin.h>
#include "../src/Position.h"

#include <iomanip>
#include <iostream>

#pragma GCC target ("bmi2,tune=skylake")

namespace engine {
    Position::Position(int sideToMove)
        : sideToMove(sideToMove)
    {
        boards[0] = 0;
        boards[1] = 0;
    }
    void Position::makeMove(Move move) {
        setHex(sideToMove==0? move: transposeMove(move),&boards[sideToMove]);
        setHex(move,&occupancy);
        sideToMove ^= 1;
        moveCount++;
    }

    void Position::unmakeMove(Move move) {
        unsetHex(sideToMove==0? move: transposeMove(move),&boards[sideToMove]);
        unsetHex(move,&occupancy);
        sideToMove ^= 1;
        moveCount--;
    }

    Move Position::getRandomLegalMove(FastRand& rng) {
        Board legal_mask = (~occupancy) & BOARD_MASK; //CHECK needed board mask? ie. can occupany ever have last 7 bits high???

        // PDEP only works on 64-bit registers, so must handle lo/hi separately
        uint64_t mask_lo = (uint64_t)legal_mask;
        uint64_t mask_hi = (uint64_t)(legal_mask >> 64);

        // Count available moves (Popcount)
        int pop_lo = __builtin_popcountll(mask_lo);
        int pop_hi = __builtin_popcountll(mask_hi);
        int total_moves = pop_lo + pop_hi;

        if (total_moves == 0) return -1;

        // Pick a random "Rank"
        // If there are 5 moves, we pick a number 0..4
        int rank = rng.range(total_moves);

        if (rank < pop_lo) {
            // CASE A: The move is in the lower 64 bits
            uint64_t sparse_selector = 1ULL << rank;
            uint64_t result = _pdep_u64(sparse_selector, mask_lo);

            // Find the index of that bit (Count Trailing Zeros)
            return __builtin_ctzll(result);

        } else {
            // CASE B: The move is in the upper 64 bits
            rank -= pop_lo;
            uint64_t sparse_selector = 1ULL << rank;
            uint64_t result = _pdep_u64(sparse_selector, mask_hi);
            // Find index and ADD 64 because we are in the upper half
            return 64 + __builtin_ctzll(result);
        }
    }

    void Position::makeRandomRolloutMove(FastRand& rng) {
        Move randomMove = getRandomLegalMove(rng);
        makeMove(randomMove);
    }

    int Position::getWinner() {
        if (moveCount == BOARD_AREA) {
            return 2;
        }
        return isWon(&boards[0])? 0: (isWon(&boards[1])? 1: -1);
    }

bool Position::isWon(Board* board) {
    // 1. Start with stones in the Top Row
    Board wavefront = *board & ROW_MASK;
    if (wavefront == 0) return false;

    // Masks to prevent wrapping
    // COL_A: 0, 11, 22...
    constexpr Board NOT_COL_A = ~COL_MASK;

    // We also need masks for "Up-Right" and "Down-Left"
    // because they shift across columns.

    Board activeStones = *board;
    Board oldWavefront = 0;

    // 2. Propagate Until Stable (Fixed Point)
    while (true) {
        oldWavefront = wavefront;

        // --- EXPAND IN ALL 6 DIRECTIONS ---

        // 1. Vertical
        Board down = (wavefront << BOARD_SIZE);
        Board up   = (wavefront >> BOARD_SIZE);

        // 2. Slanted (The tricky ones)
        // Down-Left: +10 (Needs NOT_COL_A)
        Board down_left = (wavefront & NOT_COL_A) << (BOARD_SIZE - 1);

        // Up-Right: -10 (The reverse of Down-Left).
        // When going UP, we are shifting Right relative to the array.
        // We need to ensure we don't wrap from Right Edge (Col K) to Left Edge (Col A).
        // Actually, -10 shifts bit 11 (A1) to 1 (A0). Wait.
        // Index i -> i-10.
        // If i=10 (K0), i-10=0 (A0). This wraps K->A.
        // So we need NOT_COL_A mask on the RESULT or NOT_COL_K on the SOURCE?
        // Let's invert: Down-Left shifts A -> K (bad).
        // Up-Right shifts K -> A (bad). So Up-Right needs NOT_COL_A mask on the DESTINATION
        // or NOT_COL_K mask on the SOURCE.
        // Simplest: Mask COL_A before shifting down-left. Mask COL_K before shifting up-right.
        constexpr Board NOT_COL_K = ~(COL_MASK << (BOARD_SIZE - 1));
        Board up_right = (wavefront & NOT_COL_K) >> (BOARD_SIZE - 1);

        // 3. Horizontal
        // Right: +1 (Mask COL_K)
        Board right = (wavefront & NOT_COL_K) << 1;
        // Left:  -1 (Mask COL_A)
        Board left  = (wavefront & NOT_COL_A) >> 1;

        // Combine all directions
        Board expansion = down | up | down_left | up_right | right | left;

        // Mask with stones on board
        wavefront |= (expansion & activeStones);

        // Check for convergence
        if (wavefront == oldWavefront) break;
    }

    // 3. Check Bottom Row
    return (wavefront & BOTTOM_ROW_MASK) != 0;
}

    void Position::printBitboard(Board board) const {
        std::cout << "   Raw Bitboard View:" << std::endl;

        // Header
        std::cout << "    ";
        for (int i = 0; i < BOARD_SIZE; ++i) std::cout << (char)('A' + i) << " ";
        std::cout << std::endl;

        for (int r = 0; r < BOARD_SIZE; ++r) {
            // Indent to create Hex skew
            for (int s = 0; s < r; ++s) std::cout << " ";

            // Row Number
            std::cout << std::setw(2) << (r + 1) << " ";

            for (int c = 0; c < BOARD_SIZE; ++c) {
                int index = r * BOARD_SIZE + c;
                if (hasBit(board, index)) {
                    std::cout << "1 "; // Bit is Set
                } else {
                    std::cout << ". "; // Bit is Empty
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void Position::printPosition() const {
        std::cout << "--- Game Position ---" << std::endl;
        std::cout << "Turn: " << (sideToMove==0 ? "US (X) - Vertical" : "THEM (O) - Horizontal") << std::endl;

        // Coordinate Header
        std::cout << "    ";
        for (int i = 0; i < BOARD_SIZE; ++i) std::cout << (char)('A' + i) << " ";
        std::cout << std::endl;

        for (int r = 0; r < BOARD_SIZE; ++r) {
            // 1. Skew Indentation
            for (int s = 0; s < r; ++s) std::cout << " ";

            // 2. Row Label
            std::cout << std::setw(2) << (r + 1) << " ";

            // 3. The Board Content
            for (int c = 0; c < BOARD_SIZE; ++c) {
                int physicalIndex = r * BOARD_SIZE + c;

                // We calculate the transposed index to check 'them'
                // Because 'them' thinks the board is rotated 90 degrees.
                // The physical cell (r,c) maps to index (c,r) in the transposed bitboard.
                // Note: We access the private transposeMove function here.
                Move transposedIndex = transposeMove(physicalIndex);

                bool isUs = hasBit(boards[0], physicalIndex);
                bool isThem = hasBit(boards[1], transposedIndex); // <--- Un-transpose here!

                if (isUs && isThem) {
                    std::cout << "? "; // Error state (Overlapping stones)
                } else if (isUs) {
                    // ANSI Color Red for Us
                    std::cout << "X ";
                } else if (isThem) {
                    // ANSI Color Blue for Them
                    std::cout << "Y ";
                } else {
                    std::cout << ". ";
                }
            }
            // 4. Right side connections (optional visualization aid)
            std::cout << std::endl;
        }
        auto print128 = [](std::string label, Board b) {
            uint64_t hi = (uint64_t)(b >> 64);
            uint64_t lo = (uint64_t)b;

            std::cout << label << ": ";
            if (hi > 0) {
                // Print High part, then Low part padded with leading zeros
                std::cout << "0x" << std::hex << hi
                          << "_" << std::setw(16) << std::setfill('0') << lo << std::dec;
            } else {
                // Just print Low part if High is empty
                std::cout << "0x" << std::hex << lo << std::dec;
            }
            std::cout << std::endl;
        };

        printBitboard(occupancy);

        printBitboard(boards[0]);
        printBitboard(boards[1]);

        print128("US (Vertical)   ", boards[0]);
        print128("THEM (Transposed)", boards[1]);
        print128("Occupancy       ", occupancy);
        std::cout << "ourmove? " << (sideToMove == 0);
        std::cout << "---------------------" << std::endl;
    }
} // engine