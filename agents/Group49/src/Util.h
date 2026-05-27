//
// Created by Julius on 28/11/2025.
//

#ifndef GROUP49_UTIL_H
#define GROUP49_UTIL_H
#include <array>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>

namespace engine {
    using Board = unsigned __int128;
    using Move = uint_fast8_t;

    constexpr int BOARD_SIZE = 11;
    constexpr int BOARD_AREA = BOARD_SIZE * BOARD_SIZE;
    constexpr int MAX_MOVES  = BOARD_AREA;

    const std::string MODEL_PATH = "models/best.onnx";


    constexpr Move V_START = 121;
    constexpr Move V_END   = 122;

    constexpr Board BOARD_MASK = ((Board)1 << BOARD_AREA) - 1;

    constexpr Board ROW_MASK = ((Board)1 << BOARD_SIZE) - 1;

    constexpr Board BOTTOM_ROW_MASK = ROW_MASK << ((BOARD_SIZE - 1) * BOARD_SIZE);

    constexpr Board COL_MASK = BOARD_MASK / ROW_MASK;
    constexpr Board NOT_COL_K = ~(COL_MASK << (BOARD_SIZE - 1));

    constexpr Board NOT_COL_MASK = ~COL_MASK;

    constexpr std::array<Move, BOARD_AREA> create_transpose_table() {
        std::array<Move, BOARD_AREA> table{};
        for (int i = 0; i < BOARD_AREA; ++i) {
            int row = i / BOARD_SIZE;
            int col = i % BOARD_SIZE;
            table[i] = static_cast<Move>(col * BOARD_SIZE + row);
        }
        return table;
    }

    constexpr auto TRANSPOSE_LUT = create_transpose_table();

    using NeighborList = std::array<Move, 6>;

    constexpr std::array<NeighborList, BOARD_AREA> create_neighbor_table() {
        std::array<NeighborList, BOARD_AREA> table{}; // Zero initialize

        constexpr int dr[6] = {-1, -1,  0, 0,  1, 1};
        constexpr int dc[6] = { 0,  1, -1, 1, -1, 0};

        for (int i = 0; i < BOARD_AREA; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;

            for (int k = 0; k < 6; ++k) {
                int nr = r + dr[k];
                int nc = c + dc[k];

                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    table[i][k] = static_cast<Move>(nr * BOARD_SIZE + nc);
                } else {
                    table[i][k] = 0xFF;
                }
            }
        }
        return table;
    }

    constexpr auto NEIGHBOR_LUT = create_neighbor_table();

    static inline int transposeMove(int move) {
        return TRANSPOSE_LUT[move];
    }

    constexpr std::array<Board, BOARD_AREA> create_neighbor_mask_table() {
        std::array<Board, BOARD_AREA> table{};

        constexpr int dr[6] = {-1, -1,  0, 0,  1, 1};
        constexpr int dc[6] = { 0,  1, -1, 1, -1, 0};

        for (int i = 0; i < BOARD_AREA; ++i) {
            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;
            Board mask = 0;

            for (int k = 0; k < 6; ++k) {
                int nr = r + dr[k];
                int nc = c + dc[k];

                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    int n_idx = nr * BOARD_SIZE + nc;
                    mask |= ((Board)1 << n_idx);
                }
            }
            table[i] = mask;
        }
        return table;
    }

    constexpr auto NEIGHBOR_MASKS = create_neighbor_mask_table();

    struct FastRand {
        uint64_t state;

        FastRand() {
            auto now = std::chrono::high_resolution_clock::now();
            uint64_t nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

            std::hash<std::thread::id> hasher;
            uint64_t tid = hasher(std::this_thread::get_id());

            state = nanos ^ (tid << 1) ^ 0xCAFEBABE;

            next();
            next();
            next();
        }

        explicit FastRand(uint64_t seed) : state(seed) {}

        uint64_t next() {
            uint64_t x = state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            return state = x;
        }

        uint64_t range(uint64_t max) {
            return next() % max;
        }
    };


    inline int stringToIndex(std::string coord) {
        if (coord == "swap") return -1;

        char colChar = std::tolower(coord[0]);
        int col = colChar - 'a';

        int row = std::stoi(coord.substr(1)) - 1;
        return row * BOARD_SIZE + col;
    }

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