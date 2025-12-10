//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_UTIL_H
#define GROUP49_UTIL_H
#include <array>
#include <cstdint>
#include <sstream>
#include <string>

namespace engine {
    using Board = unsigned __int128;
    using Move = uint_fast8_t;

    // 1. Basic Dimensions
    constexpr int BOARD_SIZE = 11;
    constexpr int BOARD_AREA = BOARD_SIZE * BOARD_SIZE; // 121
    constexpr int MAX_MOVES  = BOARD_AREA;

    const std::string MODEL_PATH = "/home/julius/git/hex/agents/Group49/models/hex_run_001.onnx";


    constexpr Move V_START = 121; // Top (Red) / Left (Blue)
    constexpr Move V_END   = 122; // Bottom (Red) / Right (Blue)

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

    constexpr std::array<Move, BOARD_AREA> create_transpose_table() {
        std::array<Move, BOARD_AREA> table{};
        for (int i = 0; i < BOARD_AREA; ++i) {
            int row = i / BOARD_SIZE;
            int col = i % BOARD_SIZE;
            // The logic you wrote:
            table[i] = static_cast<Move>(col * BOARD_SIZE + row);
        }
        return table;
    }

    constexpr auto TRANSPOSE_LUT = create_transpose_table();

    using NeighborList = std::array<Move, 6>;

    constexpr std::array<NeighborList, BOARD_AREA> create_neighbor_table() {
        std::array<NeighborList, BOARD_AREA> table{}; // Zero initialize

        // The 6 directions in (row, col) offsets for a skewed Hex grid
        // 1. Top       (r-1, c)
        // 2. Top-Right (r-1, c+1)
        // 3. Left      (r, c-1)
        // 4. Right     (r, c+1)
        // 5. Bot-Left  (r+1, c-1)
        // 6. Bot       (r+1, c)
        constexpr int dr[6] = {-1, -1,  0, 0,  1, 1};
        constexpr int dc[6] = { 0,  1, -1, 1, -1, 0};

        for (int i = 0; i < BOARD_AREA; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;

            for (int k = 0; k < 6; ++k) {
                int nr = r + dr[k];
                int nc = c + dc[k];

                // Check boundaries
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    table[i][k] = static_cast<Move>(nr * BOARD_SIZE + nc);
                } else {
                    table[i][k] = 0xFF; // Sentinel for "No Neighbor" (Off board)
                }
            }
        }
        return table;
    }

    // The compile-time constant
    constexpr auto NEIGHBOR_LUT = create_neighbor_table();

    static inline int transposeMove(int move) {
        return TRANSPOSE_LUT[move];
    }

    constexpr std::array<Board, BOARD_AREA> create_neighbor_mask_table() {
        std::array<Board, BOARD_AREA> table{}; // Zero initialize

        // The 6 directions in (row, col) offsets for a skewed Hex grid
        // (Same offsets as create_neighbor_table)
        constexpr int dr[6] = {-1, -1,  0, 0,  1, 1};
        constexpr int dc[6] = { 0,  1, -1, 1, -1, 0};

        for (int i = 0; i < BOARD_AREA; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;
            Board mask = 0;

            for (int k = 0; k < 6; ++k) {
                int nr = r + dr[k];
                int nc = c + dc[k];

                // Check boundaries
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    int n_idx = nr * BOARD_SIZE + nc;
                    // Cast '1' to Board (u128) to prevent overflow before shift
                    mask |= ((Board)1 << n_idx);
                }
            }
            table[i] = mask;
        }
        return table;
    }

    // The compile-time constant for Bitwise Neighbor Checks
    constexpr auto NEIGHBOR_MASKS = create_neighbor_mask_table();

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

    inline int stringToIndex(std::string coord) {
        if (coord == "swap") return -1; // Handle swap if you ever implement it

        // Lowercase string
        char colChar = std::tolower(coord[0]);

        // Handle the Go convention where 'i' is sometimes skipped.
        // MoHex/HexGui usually keeps 'i' for Hex, but be careful.
        // We will assume standard a,b,c...k mapping for now.
        int col = colChar - 'a';

        // Parse row (everything after the first char)
        int row = std::stoi(coord.substr(1)) - 1; // 1-based to 0-based

        return row * BOARD_SIZE + col;
    }

    // Converts index -> "c5"
    inline std::string indexToString(int index) {
        if (index < 0) return "resign";

        int row = index / BOARD_SIZE;
        int col = index % BOARD_SIZE;

        std::stringstream ss;
        ss << (char)('a' + col);
        ss << (row + 1);
        return ss.str();
    }
} // engine

#endif //GROUP49_UTIL_H