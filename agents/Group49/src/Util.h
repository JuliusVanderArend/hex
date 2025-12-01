//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_UTIL_H
#define GROUP49_UTIL_H
#include <array>
#include <cstdint>
namespace engine {
    using Board = unsigned __int128;
    using Move = int;//uint_fast8_t;

    // 1. Basic Dimensions
    constexpr int BOARD_SIZE = 11;
    constexpr int BOARD_AREA = BOARD_SIZE * BOARD_SIZE; // 121
    constexpr int MAX_MOVES  = BOARD_AREA;

    // 2. Board Mask (Valid bits 0 to 120)
    // Math: 2^121 - 1
    constexpr Board BOARD_MASK = ((Board)1 << BOARD_AREA) - 1;

    // 3. Row Mask (Top Row: Valid bits 0 to 10)
    // Math: 2^11 - 1 = 2047 (0x7FF)
    // CORRECTION: Changed 0xFF to 1
    constexpr Board ROW_MASK = ((Board)1 << BOARD_SIZE) - 1;

    // 4. Bottom Row Mask (Indices 110 to 120)
    // Logic: Take Top Row and shift it down 10 rows (10 * 11 = 110 bits)
    constexpr Board BOTTOM_ROW_MASK = ROW_MASK << ((BOARD_SIZE - 1) * BOARD_SIZE);

    // 5. Column A Mask (Indices 0, 11, 22, ... 110)
    // Logic: Geometric series sum (Total / Row)
    // Renamed to COL_A_MASK for clarity
    constexpr Board COL_MASK = BOARD_MASK / ROW_MASK;
    constexpr Board NOT_COL_K = ~(COL_MASK << (BOARD_SIZE - 1));

    // 6. Not Column A Mask (Used to prevent wrap-around on Down-Left shift)
    constexpr Board NOT_COL_MASK = ~COL_MASK;

    constexpr std::array<uint8_t, BOARD_AREA> create_transpose_table() {
        std::array<uint8_t, BOARD_AREA> table{};
        for (int i = 0; i < BOARD_AREA; ++i) {
            int row = i / BOARD_SIZE;
            int col = i % BOARD_SIZE;
            // The logic you wrote:
            table[i] = static_cast<uint8_t>(col * BOARD_SIZE + row);
        }
        return table;
    }

    constexpr auto TRANSPOSE_LUT = create_transpose_table();

    static inline int transposeMove(int move) {
        return TRANSPOSE_LUT[move];
    }

    struct FastRand {
        uint64_t state = 0xCAFEBABE;
        uint64_t next() {
            uint64_t x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            return state = x;
        }
        // Returns number in [0, max-1]
        uint64_t range(uint64_t max) {
            return next() % max;
        }
    };
} // engine

#endif //GROUP49_UTIL_H