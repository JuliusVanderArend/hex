#include <iomanip>
#include <gtest/gtest.h>
#include "../src/Position.h"
#include "../src/Util.h"
#include "../src/MCTS.h"

using namespace engine;

// Helper to check if Red (Player 0) has won
#define EXPECT_RED_WIN(pos) EXPECT_EQ(pos.getWinner(), 0)

// Helper to check if Blue (Player 1) has won
#define EXPECT_BLUE_WIN(pos) EXPECT_EQ(pos.getWinner(), 1)

// Helper to check if the game is still ongoing
#define EXPECT_NO_WINNER(pos) EXPECT_EQ(pos.getWinner(), -1)


// ----------------------------------------------------------------
// TEST CASES
// ----------------------------------------------------------------

// 1. Sanity Check: Empty board has no winner
TEST(WinCheck, EmptyBoard) {
    FastRand rng;
    Position pos(0);
    EXPECT_NO_WINNER(pos);
}

// 2. Simple Vertical Line (Red Win)
// Red connects Top to Bottom.
TEST(WinCheck, SimpleVerticalLine_Red) {
    FastRand rng;
    Position pos(0); // Red starts (sideToMove = 0)

    // Play down column 5
    for (int r = 0; r < BOARD_SIZE; ++r) {
        // Red moves at (r, 5)
        pos.makeMove(r * BOARD_SIZE + 5);

        // Dummy Blue move to flip turn back (at r, 0 to avoid collision)
        // Only if game isn't over (prevent infinite loops if logic is broken)
        if (pos.getWinner() == -1) {
            pos.makeMove(r * BOARD_SIZE + 0);
        }
    }

    EXPECT_RED_WIN(pos);
}

// 3. Simple Horizontal Line (Blue Win)
// Blue connects Left to Right.
// Note: Blue needs to win on *Transposed* logic usually, but getWinner handles that internally.
TEST(WinCheck, SimpleHorizontalLine_Blue) {
    FastRand rng;
    Position pos(0);

    // 1. Red plays a dummy move (0,0) so it becomes Blue's turn
    pos.makeMove(0);

    // 2. Blue plays across Row 5
    for (int c = 0; c < BOARD_SIZE; ++c) {
        // Blue moves at (5, c)
        pos.makeMove(5 * BOARD_SIZE + c);

        // Dummy Red move to flip turn back (at 0, c+1)
        if (pos.getWinner() == -1) {
             pos.makeMove(c + 1);
        }
    }



    EXPECT_BLUE_WIN(pos);
}

// 4. The "Wrap-Around" Bug Test (CRITICAL)
// Ensures bitshifts don't teleport from Left Edge to Right Edge.
TEST(WinCheck, NoWrapAround) {
    FastRand rng;
    Position pos(0);

    /*
     * Setup Red Stones:
     * (0, 0)  -> Top Left corner
     * (1, 10) -> Right edge of the row below
     * * If COL_A_MASK logic is missing, the "Down-Left" shift from (0,0)
     * might wrap around to (1,10), creating a false connection.
     */

    pos.makeMove(0);   // Red at (0,0)
    pos.makeMove(50);  // Blue dummy

    pos.makeMove(21);  // Red at (1, 10) [Index 21 = 1*11 + 10]
    pos.makeMove(51);  // Blue dummy

    // This should NOT be a win.
    EXPECT_NO_WINNER(pos);
}

// 5. A Zig-Zag "Snake" Path (Red Win)
// Tests that the wave propagates correctly through complex connections
TEST(WinCheck, SnakePath_Red) {
    FastRand rng;
    Position pos(0);

    // A path that wiggles down the board
    std::vector<int> redIndices = {
        0*11+1, 1*11+1, 1*11+0, 2*11+0, 2*11+1, // Zig-zag top
        3*11+1, 3*11+2, 4*11+2,                 // Middle
        5*11+2, 6*11+2, 6*11+1, 7*11+1,         // Middle-Low
        8*11+1, 9*11+1, 10*11+1                 // Bottom connection
    };

    for (int idx : redIndices) {
        pos.makeMove(idx); // Red move

        // Play dummy blue move safely away (e.g., column 10)
        if (pos.getWinner() == -1) {
            pos.makeMove(idx + 5); // Just some offset to avoid collision
        }
        pos.printPosition();
    }
    pos.printPosition();
    EXPECT_RED_WIN(pos);
}

// 6. The "Almost Won" (Broken Line)
// Tests that the engine doesn't return early positives.
TEST(WinCheck, BrokenLine_Red) {
    FastRand rng;
    Position pos(0);

    // Vertical line down column 2, but MISSING row 5
    for (int r = 0; r < BOARD_SIZE; ++r) {
        if (r == 5) continue; // THE GAP

        pos.makeMove(r * BOARD_SIZE + 2); // Red
        pos.makeMove(r * BOARD_SIZE + 8); // Blue dummy
    }

    EXPECT_NO_WINNER(pos);
}

TEST(WinCheck, UTurnPath_Red) {
    FastRand rng;
    Position pos(0);

    // THE SHAPE:
    // A path that goes down Col 0, hits a wall,
    // hooks RIGHT, then UP, then RIGHT, then DOWN.
    //
    // x . .    (3,0)
    // x x x    (4,0) -> (4,1) [UP-TURN] -> (4,2)
    // x x .    (5,0) -> (5,1) [Bottom of U]
    // . . x    (6,2)

    std::vector<std::pair<int, int>> redMoves = {
        // 1. The Left Wall (Down Column 0)
        {0,0}, {1,0}, {2,0}, {3,0}, {4,0}, {5,0},

        // 2. The Bottom of the U (Move Right)
        {5,1},

        // 3. The CRITICAL UP-MOVE (The U-Turn)
        // From (5,1), we must go UP to (4,1) to continue.
        // Note: (5,1) connects to (4,1) in Hex.
        {4,1},

        // 4. The Escape (Move Right)
        {4,2},

        // 5. The Path to Victory (Down Column 2)
        {5,2}, {6,2}, {7,2}, {8,2}, {9,2}, {10,2}
    };

    // Apply moves
    for (auto p : redMoves) {
        int idx = p.first * BOARD_SIZE + p.second;
        pos.makeMove(idx); // Red

        // Dummy Blue move to maintain turn order
        if (pos.getWinner() == -1) {
            // Pick a safe spot far away (Col 10) to ensure no interference
            pos.makeMove(p.first * BOARD_SIZE + 10);
        }
    }

    // Diagnostic: If this fails, uncomment the print to see where the wave stopped
    // pos.debugWinCheck();

    EXPECT_RED_WIN(pos);
}

TEST(WinCheck, BrokenBucketPath_NoWin) {
    FastRand rng;
    Position pos(0);

    /* * SCENARIO: The "Broken Apex"
     * Red builds a U-shape that goes DOWN Col 0, UP Col 2,
     * and tries to connect to a separate group going DOWN Col 4.
     * * The connection depends on a single stone at the apex (top of the curve).
     * We purposefully OMIT that stone.
     */

    // 1. LEFT TOWER (Connected to Top)
    // Path: (0,0) down to (5,0), then U-Turn up to (4,2)
    std::vector<int> leftTower = {
        // Down Column 0 (Connected to Top)
        0*11+0, 1*11+0, 2*11+0, 3*11+0, 4*11+0, 5*11+0,

        // The "Bottom" of the bucket (Move Right)
        5*11+1,

        // The "Right Side" of the bucket (Move UP)
        5*11+2, 4*11+2
    };

    // 2. RIGHT TOWER (Connected to Bottom)
    // Path: Starts at (4,4) and goes down to (10,4)
    std::vector<int> rightTower = {
        4*11+4, // Apex of right tower
        5*11+4, 6*11+4, 7*11+4, 8*11+4, 9*11+4, 10*11+4
    };

    // 3. APPLY MOVES
    for (int move : leftTower) {
        pos.makeMove(move);
        pos.makeMove(move + 50); // Dummy blue move to avoid self-play error
    }
    for (int move : rightTower) {
        pos.makeMove(move);
        pos.makeMove(move + 50); // Dummy blue move
    }

    // 4. THE GAP ANALYSIS
    // Left Tower ends at (4,2).
    // Right Tower starts at (4,4).
    // The "Keystone" would be (3,3) or (4,3).
    // We check that NO WIN is detected without these stones.

    // Uncomment to see the gap visually:
    // pos.printPosition();

    EXPECT_NO_WINNER(pos);
}

class ConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if model exists to avoid crashing tests hard
        if (access(MODEL_PATH.c_str(), F_OK) == -1) {
            std::cerr << "[WARNING] Model not found at " << MODEL_PATH
                      << ". Consistency tests using MCTS will fail/skip." << std::endl;
        }
    }
};

// TEST 1: MCTS vs MCTS (Realistic Game Paths)
TEST_F(ConsistencyTest, EngineSelfPlay_NoCorruption) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();

    // 1. Setup Inference Stack
    Inference net(MODEL_PATH);
    InferenceServer server(net);
    MCTS agent;

    for (int game = 0; game <5; ++game) {
        Position pos(0);
        int moves = 0;

        while (pos.getWinner() == -1) {
            pos.checkConsistency();

            // 2. Run MCTS
            // 200 iterations is enough for a sanity check
            SearchResult result = agent.searchWithPolicy(pos, server, 200);
            int bestMove = result.bestMove;

            // 3. ASSERT LEGALITY
            ASSERT_TRUE(pos.isMoveLegal(bestMove))
                << "FATAL: MCTS returned illegal move " << bestMove
                << " at move count " << moves;

            // 4. Make Move
            pos.makeMove(bestMove);
            moves++;

            pos.checkConsistency();

            if (moves > BOARD_AREA) break;
        }
        std::cout << moves << " moves in game " << game << std::endl;
        EXPECT_NE(pos.getWinner(), -1) << "Game " << game << " did not finish.";
    }
}

// TEST 2: Random vs Random (Chaos / Edge Cases)
// No NN needed here, relies on internal random logic
TEST_F(ConsistencyTest, RandomChaos_NoCorruption) {
    FastRand rng;

    // Run 50 fast random games
    for (int game = 0; game < 50; ++game) {
        Position pos(0);
        int moves = 0;

        while (pos.getWinner() == -1) {
            pos.checkConsistency();

            // Pure random selection
            int move = pos.getRandomLegalMove(rng);

            if (move == -1) break;

            ASSERT_TRUE(pos.isMoveLegal(move))
                << "FATAL: getRandomLegalMove returned occupied square " << move;

            pos.makeMove(move);
            moves++;

            pos.checkConsistency();
        }
    }
}

class PerformanceTest : public ::testing::Test {};

// TEST: Benchmark MCTS Performance (Nodes Per Second)
TEST_F(PerformanceTest, Calculate_NPS) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();

    // 1. Setup
    Inference net(MODEL_PATH);
    InferenceServer server(net);
    Position pos(0);
    MCTS agent;

    // Configuration:
    // NN Inference is slow. 500 iterations is enough to gauge NPS.
    int iterations = 500000;

    std::cout << "[Benchmark] Starting Neural MCTS Search (" << iterations << " iterations)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // 2. Run Search
    agent.searchWithPolicy(pos, server, iterations);

    auto end = std::chrono::high_resolution_clock::now();

    // 3. Calculate Metrics
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (duration_ms == 0) duration_ms = 1;

    double seconds = duration_ms / 1000.0;
    int nps = (int)(iterations / seconds);

    std::cout << "============================================" << std::endl;
    std::cout << " Total Time: " << duration_ms << " ms" << std::endl;
    std::cout << " Speed:      " << nps << " Nodes/Sec (NPS)" << std::endl;
    std::cout << "============================================" << std::endl;

    // 4. Quality Assertion
    // NN MCTS is much slower than random rollouts.
    // 50 NPS is a reasonable baseline for CPU inference in WSL.
    // If you use GPU/TensorRT later, expect 500-2000+.
    EXPECT_GT(nps, 50) << "Performance is dangerously low (check Batching logic).";
}

class OnnxTest : public ::testing::Test {
protected:
    void printTopMoves(const std::vector<float>& policy, int topK = 5) {
        std::vector<std::pair<int, float>> moves;
        for (int i = 0; i < policy.size(); ++i) {
            moves.push_back({i, policy[i]});
        }
        std::sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::cout << "--- Top " << topK << " Policy Moves ---" << std::endl;
        for (int i = 0; i < topK && i < moves.size(); ++i) {
            int moveIdx = moves[i].first;
            float prob = moves[i].second;
            int r = moveIdx / BOARD_SIZE;
            int c = moveIdx % BOARD_SIZE;
            char colChar = 'A' + c;

            std::cout << std::setw(2) << (i + 1) << ". "
                      << colChar << (r + 1) << " (" << moveIdx << ") "
                      << " -> " << std::fixed << std::setprecision(4) << prob
                      << std::endl;
        }
    }
};

TEST_F(OnnxTest, RandomSelfPlay_Inference) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();

    // Use direct Inference class (no Server needed for single-shot tests)
    Inference net(MODEL_PATH);
    Position pos(0);
    FastRand rng;

    // 2. PLAY SOME MOVES
    int movesToPlay = 8;
    for (int i = 0; i < movesToPlay; ++i) {
        pos.makeRandomRolloutMove(rng);
    }

    pos.printPosition();

    // 3. RUN INFERENCE (Single)
    std::cout << "Running ONNX Inference..." << std::endl;
    auto result = net.predict(pos);

    std::vector<float> policy = result.first;
    float value = result.second;

    std::cout << "Predicted Value: " << value << std::endl;
    printTopMoves(policy);

    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
    EXPECT_EQ(policy.size(), 121);
}

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "../src/Position.h"
#include "../src/Util.h" // For BOARD_SIZE

using namespace engine;

class StrategyTest : public ::testing::Test {
protected:
    // Helper: Convert (Row, Col) to index 0-120
    int idx(int row, int col) {
        return row * BOARD_SIZE + col;
    }

    // Helper: Print readable ranks for debugging
    void verifyMoveRank(const std::vector<float>& policy, int targetMove, std::string moveName, int maxRank = 5) {
        int rank = 1;
        for (int i = 0; i < BOARD_AREA; ++i) {
            if (policy[i] > policy[targetMove]) rank++;
        }

        std::cout << "Move " << moveName << " (Index " << targetMove << ") Rank: #" << rank
                  << " (Prob: " << std::fixed << std::setprecision(4) << policy[targetMove] << ")" << std::endl;

        EXPECT_LE(rank, maxRank) << "Network failed to identify " << moveName << " as a top candidate.";
    }
};

// 1. RED OFFENSE: Can Red see a simple Vertical win?
TEST_F(StrategyTest, Red_ImmediateWin_Vertical) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();
    Inference net(MODEL_PATH);
    Position pos(0); // Red to move

    // Setup: Red has a column going down F (Col 5), missing only the bottom stone.
    // Red: F1..F10 (Rows 0-9, Col 5)
    // Blue: Garbage moves (A1..A10)
    for (int row = 0; row < 10; ++row) {
        pos.makeMove(idx(row, 5)); // Red F(row)
        pos.makeMove(idx(row, 0)); // Blue A(row) - useless
    }

    // TARGET: F11 (Row 10, Col 5) to win
    int winningMove = idx(10, 5);

    auto result = net.predict(pos);
    float value = result.second;

    // Expectations
    // 1. Value should be very high (Red is winning)
    EXPECT_GT(value, 0.5f) << "Red should be winning clearly";

    // 2. Policy should prioritize the winning move
    verifyMoveRank(result.first, winningMove, "F11 (Vertical Win)");
}

// 2. BLUE OFFENSE: Can Blue see a simple Horizontal win?
// This verifies that your Input Tensor Transposition logic is working!
TEST_F(StrategyTest, Blue_ImmediateWin_Horizontal) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();
    Inference net(MODEL_PATH);
    Position pos(0);

    // Setup: Blue has a row across Row 5, missing the last stone.
    // We need it to be BLUE's turn, so Red plays garbage first.
    pos.makeMove(idx(0, 0)); // Red garbage (A1)

    // Blue: A6..J6 (Row 5, Cols 0-9)
    // Red:  B1..K1 (Row 0, Cols 1-10) - useless
    for (int col = 0; col < 10; ++col) {
        pos.makeMove(idx(5, col)); // Blue Row 5
        pos.makeMove(idx(0, col + 1)); // Red Row 0
    }

    // Now it is BLUE's turn.
    // TARGET: K6 (Row 5, Col 10) to connect Left-Right
    int winningMove = idx(5, 10);

    auto result = net.predict(pos);
    float value = result.second;

    // Expectations
    // 1. Value should be LOW (from Red's perspective, Red is losing)
    // OR HIGH if your net outputs value relative to "SideToMove".
    // Assuming standard AlphaZero (Value is always for current player):
    // If output is SideToMove perspective: > 0.5
    // If output is Always Red perspective: < -0.5
    // Let's assume SideToMove (standard for self-play nets):
    EXPECT_GT(value, 0.5f) << "Blue (Current Player) should be winning";

    // 2. Policy should prioritize K6
    verifyMoveRank(result.first, winningMove, "K6 (Horizontal Win)");
}

// 3. RED DEFENSE: Can Red block a Blue win?
TEST_F(StrategyTest, Red_MustBlock_Horizontal) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();
    Inference net(MODEL_PATH);
    Position pos(0);

    // Setup: Blue is threatening to win on Row 5.
    // Red must play there to stop it.

    // Blue: A6..J6 (Row 5, Cols 0-9)
    // Red: Garbage elsewhere
    for (int col = 0; col < 10; ++col) {
        pos.makeMove(idx(0, col)); // Red garbage
        pos.makeMove(idx(5, col)); // Blue threat
    }

    // It is RED's turn. Blue has 0-9 filled in Row 5.
    // If Red doesn't play K6 (Row 5, Col 10), Blue wins next turn.
    int blockingMove = idx(5, 10);

    auto result = net.predict(pos);
    float value = result.second;

    // Expectations
    // 1. Value might be low (Red is in danger), but shouldn't be -1.0 yet if the block saves it.
    std::cout << "Red Defense Value: " << value << std::endl;

    // 2. The Blocking move MUST be top priority
    verifyMoveRank(result.first, blockingMove, "K6 (Block Horizontal)", 3);
}

// 4. BLUE DEFENSE: Can Blue block a Red win?
TEST_F(StrategyTest, Blue_MustBlock_Vertical) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();
    Inference net(MODEL_PATH);
    Position pos(0);

    // Setup: Red is threatening vertical win on Col 5.
    // Red: F1..F10 (Rows 0-9, Col 5)
    // Blue: Garbage
    for (int row = 0; row < 10; ++row) {
        pos.makeMove(idx(row, 5)); // Red threat
        pos.makeMove(idx(row, 0)); // Blue garbage
    }

    // Make one more Red move to pass turn to Blue?
    // No, loop played 10 moves each. It is RED's turn now.
    // We need BLUE to move. Red plays one more useless move.
    pos.makeMove(idx(0, 1));

    // Now Blue's Turn. Red has Col 5 almost filled.
    // Blue must play F11 (Row 10, Col 5) to block.
    int blockingMove = idx(10, 5);

    auto result = net.predict(pos);

    verifyMoveRank(result.first, blockingMove, "F11 (Block Vertical)", 3);
}

// 5. OPENING: Does the network understand the center is best?
TEST_F(StrategyTest, Opening_CenterBias) {
    if (access(MODEL_PATH.c_str(), F_OK) == -1) GTEST_SKIP();
    Inference net(MODEL_PATH);
    Position pos(0); // Empty board

    // Target: Center Hex (F6 -> Row 5, Col 5)
    int center = idx(5, 5);

    auto result = net.predict(pos);
    float value = result.second;

    // 1. Value should be slightly positive (First player advantage in Hex)
    EXPECT_GT(value, 0.0f);
    EXPECT_LT(value, 0.3f); // Shouldn't be a forced win yet!

    // 2. Center should be highly ranked
    verifyMoveRank(result.first, center, "F6 (Center)", 10);

    // 3. Ensure corner (A1) is NOT highly ranked
    int corner = idx(0, 0);
    int cornerRank = 1;
    for (float p : result.first) if (p > result.first[corner]) cornerRank++;

    EXPECT_GT(cornerRank, 20) << "Network shouldn't prefer corners on move 1";
}