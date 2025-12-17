//
// Created by Julius on 28/11/2025.
//
#include "../src/Position.h"

#include <iomanip>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#pragma GCC target ("bmi2,tune=skylake")
#else
// Software fallback for PDEP on ARM/Apple Silicon
static inline uint64_t _pdep_u64(uint64_t val, uint64_t mask) {
    uint64_t result = 0;
    for (uint64_t bit = 1; mask; bit += bit) {
        if (val & bit)
            result |= mask & -mask;
        mask &= mask - 1;
    }
    return result;
}
#endif

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
                // Check bitboard: Is this neighbor occupied by us?
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
        moves.reserve(121 - moveCount); // Pre-allocate memory

        // Invert occupancy to get empty spots
        // We process the board in 64-bit chunks for speed
        Board empty = (~occupancy) & BOARD_MASK;

        uint64_t lo = (uint64_t)empty;
        uint64_t hi = (uint64_t)(empty >> 64);

        // Scan Lower 64 bits
        while (lo) {
            int idx = __builtin_ctzll(lo);
            moves.push_back(idx);
            lo &= (lo - 1); // Clear lowest bit
        }

        // Scan Upper 64 bits
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
            // CASE A: The move is in the lower 64 bits
            uint64_t sparse_selector = 1ULL << rank;
            uint64_t result = _pdep_u64(sparse_selector, mask_lo);

            // Find the index of move
            return __builtin_ctzll(result);

        } else {
            // CASE B: The move is in the upper 64 bits
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
    // ------------------------------------------------------------------------
    // PHASE 1: Generate Random Move (PDEP)
    // ------------------------------------------------------------------------

    // 1. Calculate available moves
    // Note: We use the inverted occupancy to find holes
    Board legal_mask = (~occupancy) & BOARD_MASK;

    // 2. Split into 64-bit halves for PDEP (Instruction Set Constraint)
    uint64_t mask_lo = (uint64_t)legal_mask;
    uint64_t mask_hi = (uint64_t)(legal_mask >> 64);

    // 3. Count available moves
    int pop_lo = __builtin_popcountll(mask_lo);
    int pop_hi = __builtin_popcountll(mask_hi);
    int total_moves = pop_lo + pop_hi;

    // Safety: If no moves left, return (Caller should handle isFull/isWon)
    if (total_moves == 0) return;

    // 4. Select Random Rank
    int rank = rng.range(total_moves);

    // 5. PDEP: Map the Random Rank to a Physical Board Index
    Move move_idx;       // Raw physical index (0-120)
    Board move_bit;      // The bitmask for this move

    if (rank < pop_lo) {
        // Move is in the lower 64 bits
        uint64_t sparse = 1ULL << rank;
        uint64_t res = _pdep_u64(sparse, mask_lo);

        move_idx = __builtin_ctzll(res);
        move_bit = (Board)res;
    } else {
        // Move is in the upper 64 bits
        rank -= pop_lo;
        uint64_t sparse = 1ULL << rank;
        uint64_t res = _pdep_u64(sparse, mask_hi);

        move_idx = 64 + __builtin_ctzll(res);
        move_bit = ((Board)res) << 64;
    }

    // ------------------------------------------------------------------------
    // PHASE 2: State Updates (Inlined)
    // ------------------------------------------------------------------------

    // 1. Update Global Occupancy
    // We use the pre-calculated move_bit to avoid a shift operation
    occupancy |= move_bit;

    // 2. Determine Canonical (Transposed) Index
    // Red (0) = Raw Index. Blue (1) = Transposed Index.
    // Optimization: Branchless lookup is often possible, but 'if' is fine here.
    Move canonical_idx;
    if (sideToMove == 0) {
        canonical_idx = move_idx;
        // Optimization: Red's board matches physical, so re-use move_bit
        boards[0] |= move_bit;
    } else {
        canonical_idx = TRANSPOSE_LUT[move_idx];
        // Must calculate transposed bitmask.
        // Note: Casting '1' to Board (u128) is critical.
        boards[1] |= ((Board)1 << canonical_idx);
    }

    // ------------------------------------------------------------------------
    // PHASE 3: DSU Connectivity (The Critical Optimization)
    // ------------------------------------------------------------------------

    // We update ONLY the current player's DSU
    HexDSU& dsu = dsus[sideToMove];

    // 1. Edge Connections (Virtual Nodes)
    // V_START = 121, V_END = 122
    if (canonical_idx < BOARD_SIZE) {
        dsu.unite(canonical_idx, V_START);
    }
    else if (canonical_idx >= BOARD_AREA - BOARD_SIZE) {
        dsu.unite(canonical_idx, V_END);
    }

    // 2. Neighbor Connections (Bitwise Intersection)
    // Instead of looping 6 times, we intersect our board with the neighbor mask.
    // This gives us a bitboard of ONLY the existing friendly neighbors.
    Board neighbors = boards[sideToMove] & NEIGHBOR_MASKS[canonical_idx];

    // Iterate while there are still neighbor bits set
    while (neighbors) {
        // Find the index of the next neighbor
        // Note: We need a u128 safe ctz.
        // Since we can't easily do __builtin_ctz128, we handle lo/hi split.

        uint64_t n_lo = (uint64_t)neighbors;
        int n_idx;

        if (n_lo) {
            n_idx = __builtin_ctzll(n_lo);
            // Clear the bit locally to advance the loop
            // Optimization: x & (x-1) clears the lowest set bit
            neighbors &= (neighbors - 1);
            // NOTE: If neighbors was pure u128, the subtraction handles carry.
            // But since we operate on 'neighbors' (u128) in the condition,
            // the subtract works correctly across the boundary.
        } else {
            // Neighbor is in the high 64 bits
            uint64_t n_hi = (uint64_t)(neighbors >> 64);
            n_idx = 64 + __builtin_ctzll(n_hi);

            // Clear the bit (high part subtraction)
             neighbors &= (neighbors - 1);
        }

        // Connect!
        dsu.unite(canonical_idx, n_idx);
    }

    // ------------------------------------------------------------------------
    // PHASE 4: Finalize
    // ------------------------------------------------------------------------
    sideToMove ^= 1;
    moveCount++;
}

    int Position::getWinner() const {
        // Check Red (0)
        if (sideToMove ==1) {
            if (dsus[0].isConnected(V_START, V_END)) return 0;
        }

        // Check Blue (1)
        if (sideToMove ==0) {
            if (dsus[1].isConnected(V_START, V_END)) return 1;
        }

        // Draw / Ongoing
        // if (moveCount == BOARD_AREA) return 2; // Should technically never happen in Hex if logic is perfect
        return -1;
    }


    void Position::toTensor(float* dst) const {
        constexpr int PLANE_SIZE = 121;

        float* ptr = dst;

        // --- 1. Red Stones ---
        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = ((boards[0] >> i) & 1) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        // --- 2. Blue Stones (transposed) ---
        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = ((boards[1] >> transposeMove(i)) & 1) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        // --- 3. Turn (side to move) ---
        float turnVal = (sideToMove == 0) ? 1.0f : 0.0f;
        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = turnVal;
        ptr += PLANE_SIZE;

        // --- 4. Last move ---
        for (int i = 0; i < PLANE_SIZE; ++i)
            ptr[i] = (i == lastMove) ? 1.0f : 0.0f;
        ptr += PLANE_SIZE;

        // --- 5. Connected to V_START ---
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

        // --- 6. Connected to V_END ---
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
        toTensor(tensor.data());   // <-- используем новый метод
        return tensor;
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

    bool Position::hasStone(int player, Move move) const {
        if (player == 0) {
            return hasBit(boards[0], move);
        }
        Move transposed = transposeMove(move);
        return hasBit(boards[1], transposed);
    }
} // engine
