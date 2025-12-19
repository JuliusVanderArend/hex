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
        occupancy = 0;
        dsus[0].reset();
        dsus[1].reset();
        lastMove = 1;
    }
    void Position::makeMove(Move move) {
        lastMove = move;
        Move canonicalMove = sideToMove==0? move: transposeMove(move);
        setHex(canonicalMove,&boards[sideToMove]);
        if (canonicalMove < BOARD_SIZE) {
            dsus[sideToMove].unite(canonicalMove,V_START); // should unite with absolute or cannonical move???
        }
        else if (canonicalMove >= BOARD_AREA - BOARD_SIZE) { // >= is correct????
            dsus[sideToMove].unite(canonicalMove,V_END);
        }
        setHex(move,&occupancy);

        const auto& myNeighbors = NEIGHBOR_LUT[canonicalMove];

        for (int i = 0; i < 6; ++i) {
            int n_idx = myNeighbors[i];
            if (n_idx != -1) {
                if ((boards[sideToMove] >> n_idx) & 1) {
                    dsus[sideToMove].unite(canonicalMove, n_idx);
                }
            }
        }
        sideToMove ^= 1;
        moveCount++; //needed ?
    }

    void Position::unmakeMove(Move move) {
        unsetHex(sideToMove==0? move: transposeMove(move),&boards[sideToMove]);
        unsetHex(move,&occupancy);
        sideToMove ^= 1;
        moveCount--;
    }

    std::vector<int> Position::getLegalMoves() const {
        std::vector<int> moves;
        moves.reserve(121 - moveCount);

        // Invert occupancy to get empty spots
        Board empty = (~occupancy) & BOARD_MASK;

        uint64_t lo = (uint64_t)empty;
        uint64_t hi = (uint64_t)(empty >> 64);

        while (lo) {
            int idx = __builtin_ctzll(lo);
            moves.push_back(idx);
            lo &= (lo - 1); // Clear lowest bit
        }

        while (hi) {
            int idx = 64 + __builtin_ctzll(hi);
            moves.push_back(idx);
            hi &= (hi - 1);
        }
        return moves;
    }

    Move Position::getRandomLegalMove(FastRand& rng) const{
        Board legal_mask = (~occupancy) & BOARD_MASK; //CHECK needed board mask? ie. can occupany ever have last 7 bits high???

        // split upper and lower board so it works with 64 bit PDEP
        uint64_t mask_lo = (uint64_t)legal_mask;
        uint64_t mask_hi = (uint64_t)(legal_mask >> 64);

        // Count available moves
        int pop_lo = __builtin_popcountll(mask_lo);
        int pop_hi = __builtin_popcountll(mask_hi);
        int total_moves = pop_lo + pop_hi;

        if (total_moves == 0) return -1;

        // Pick a random move from avaliable moves remaining
        int rank = rng.range(total_moves);

        if (rank < pop_lo) {
            uint64_t sparse_selector = 1ULL << rank;
            uint64_t result = _pdep_u64(sparse_selector, mask_lo);

            // Find the index of move
            return __builtin_ctzll(result);

        } else {
            rank -= pop_lo;
            uint64_t sparse_selector = 1ULL << rank;
            uint64_t result = _pdep_u64(sparse_selector, mask_hi);
            return 64 + __builtin_ctzll(result);
        }
    }

    // void Position::makeRandomRolloutMove(FastRand& rng) {
    //     Move legalMove = getRandomLegalMove(rng);
    //     makeMove(legalMove);
    // }
void Position::makeRandomRolloutMove(FastRand& rng) {
    Board legal_mask = (~occupancy) & BOARD_MASK;

    uint64_t mask_lo = (uint64_t)legal_mask;
    uint64_t mask_hi = (uint64_t)(legal_mask >> 64);

    int pop_lo = __builtin_popcountll(mask_lo);
    int pop_hi = __builtin_popcountll(mask_hi);
    int total_moves = pop_lo + pop_hi;

    if (total_moves == 0) return;

    int rank = rng.range(total_moves);

    Move move_idx;
    Board move_bit;

    if (rank < pop_lo) {
        uint64_t sparse = 1ULL << rank;
        uint64_t res = _pdep_u64(sparse, mask_lo);

        move_idx = __builtin_ctzll(res);
        move_bit = (Board)res;
    } else {
        rank -= pop_lo;
        uint64_t sparse = 1ULL << rank;
        uint64_t res = _pdep_u64(sparse, mask_hi);

        move_idx = 64 + __builtin_ctzll(res);
        move_bit = ((Board)res) << 64;
    }

    occupancy |= move_bit;

    Move canonical_idx;
    if (sideToMove == 0) {
        canonical_idx = move_idx;
        boards[0] |= move_bit;
    } else {
        canonical_idx = TRANSPOSE_LUT[move_idx];
        boards[1] |= ((Board)1 << canonical_idx);
    }

    HexDSU& dsu = dsus[sideToMove];

    if (canonical_idx < BOARD_SIZE) {
        dsu.unite(canonical_idx, V_START);
    }
    else if (canonical_idx >= BOARD_AREA - BOARD_SIZE) {
        dsu.unite(canonical_idx, V_END);
    }

    Board neighbors = boards[sideToMove] & NEIGHBOR_MASKS[canonical_idx];

    while (neighbors) {
        uint64_t n_lo = (uint64_t)neighbors;
        int n_idx;

        if (n_lo) {
            n_idx = __builtin_ctzll(n_lo);
            neighbors &= (neighbors - 1);
        } else {
            uint64_t n_hi = (uint64_t)(neighbors >> 64);
            n_idx = 64 + __builtin_ctzll(n_hi);
            neighbors &= (neighbors - 1);
        }

        dsu.unite(canonical_idx, n_idx);
    }

    sideToMove ^= 1;
    moveCount++;
}

    int Position::getWinner() const {
        if (sideToMove ==1) {
            if (dsus[0].isConnected(V_START, V_END)) return 0;
        }

        if (sideToMove ==0) {
            if (dsus[1].isConnected(V_START, V_END)) return 1;
        }

        return -1;
    }

    void Position::loadFromSnapshot(const std::vector<std::string>& rows) {
    boards[0] = 0;
    boards[1] = 0;
    occupancy = 0;
    moveCount = 0;
    lastMove = -1;

    dsus[0].reset();
    dsus[1].reset();

    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            char c = rows[y][x];
            int idx = y * BOARD_SIZE + x;

            if (c == 'R') {
                boards[0] |= ((Board)1 << idx);
                occupancy |= ((Board)1 << idx);
                moveCount++;
            }
            else if (c == 'B') {
                int t = x * BOARD_SIZE + y; // transpose
                boards[1] |= ((Board)1 << t);
                occupancy |= ((Board)1 << idx);
                moveCount++;
            }
        }
    }

    static const int dr[6] = {-1, -1, 0, 0, 1, 1};
    static const int dc[6] = {0, 1, -1, 1, -1, 0};

    for (int player = 0; player < 2; player++) {
        for (int i = 0; i < BOARD_AREA; i++) {
            if (!((boards[player] >> i) & 1)) continue;

            int r = i / BOARD_SIZE;
            int c = i % BOARD_SIZE;

            for (int d = 0; d < 6; d++) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE)
                    continue;

                int ni = nr * BOARD_SIZE + nc;
                if ((boards[player] >> ni) & 1) {
                    dsus[player].unite(i, ni);
                }
            }

            if (player == 0) {
                if (r == 0) dsus[0].unite(i, 121);
                if (r == BOARD_SIZE - 1) dsus[0].unite(i, 122);
            } else { // Blue
                if (c == 0) dsus[1].unite(i, 121);
                if (c == BOARD_SIZE - 1) dsus[1].unite(i, 122);
            }
        }
    }

    sideToMove = (moveCount % 2 == 0) ? 0 : 1;
}


    void Position::toTensor(float* dst) const {
        constexpr int PLANE_SIZE = 121;

        float* ptr = dst;

        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = ((boards[0] >> i) & 1) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = ((boards[1] >> transposeMove(i)) & 1) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        float turnVal = (sideToMove == 0) ? 1.0f : 0.0f;
        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = turnVal;
        ptr += PLANE_SIZE;

        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = (i == lastMove) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        for (int i = 0; i < PLANE_SIZE; ++i) {
            bool isRed  = (boards[0] >> i) & 1;
            bool isBlue = (boards[1] >> transposeMove(i)) & 1;

            bool connected = false;
            if (isRed)
                connected = dsus[0].isConnected(i, V_START);
            else if (isBlue)
                connected = dsus[1].isConnected(transposeMove(i), V_START);

            ptr[i] = connected ? 1.0f : 0.0f;
        }
        ptr += PLANE_SIZE;

        for (int i = 0; i < PLANE_SIZE; ++i) {
            bool isRed  = (boards[0] >> i) & 1;
            bool isBlue = (boards[1] >> transposeMove(i)) & 1;

            bool connected = false;
            if (isRed)
                connected = dsus[0].isConnected(i, V_END);
            else if (isBlue)
                connected = dsus[1].isConnected(transposeMove(i), V_END);

            ptr[i] = connected ? 1.0f : 0.0f;
        }
    }


    std::vector<float> Position::toTensor() const {
        constexpr int CHANNELS = 6;
        constexpr int PLANE_SIZE = 121;

        std::vector<float> tensor(CHANNELS * PLANE_SIZE);
        toTensor(tensor.data());
        return tensor;
    }

    void Position::printBitboard(Board board) const {
        std::cout << "   Raw Bitboard View:" << std::endl;

        std::cout << "    ";
        for (int i = 0; i < BOARD_SIZE; ++i) std::cout << (char)('A' + i) << " ";
        std::cout << std::endl;

        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int s = 0; s < r; ++s) std::cout << " ";

            std::cout << std::setw(2) << (r + 1) << " ";

            for (int c = 0; c < BOARD_SIZE; ++c) {
                int index = r * BOARD_SIZE + c;
                if (hasBit(board, index)) {
                    std::cout << "1 ";
                } else {
                    std::cout << ". ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void Position::printPosition() const {
        std::cout << "--- Game Position ---" << std::endl;
        std::cout << "Turn: " << (sideToMove==0 ? "US (X) - Vertical" : "THEM (O) - Horizontal") << std::endl;

        std::cout << "    ";
        for (int i = 0; i < BOARD_SIZE; ++i) std::cout << (char)('A' + i) << " ";
        std::cout << std::endl;

        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int s = 0; s < r; ++s) std::cout << " ";

            std::cout << std::setw(2) << (r + 1) << " ";

            for (int c = 0; c < BOARD_SIZE; ++c) {
                int physicalIndex = r * BOARD_SIZE + c;

                Move transposedIndex = transposeMove(physicalIndex);

                bool isUs = hasBit(boards[0], physicalIndex);
                bool isThem = hasBit(boards[1], transposedIndex); // <--- Un-transpose here!

                if (isUs && isThem) {
                    std::cout << "? ";
                } else if (isUs) {
                    std::cout << "X ";
                } else if (isThem) {
                    std::cout << "Y ";
                } else {
                    std::cout << ". ";
                }
            }
            std::cout << std::endl;
        }
        auto print128 = [](std::string label, Board b) {
            uint64_t hi = (uint64_t)(b >> 64);
            uint64_t lo = (uint64_t)b;

            std::cout << label << ": ";
            if (hi > 0) {
                std::cout << "0x" << std::hex << hi
                          << "_" << std::setw(16) << std::setfill('0') << lo << std::dec;
            } else {
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

    bool Position::hasStone(int player, Move move) const {
        if (player == 0) {
            return hasBit(boards[0], move);
        }
        Move transposed = transposeMove(move);
        return hasBit(boards[1], transposed);
    }
} // engine
