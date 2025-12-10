#include <iomanip>
#include <gtest/gtest.h>
#include "../src/Position.h"
#include "../src/Util.h"
#include "../src/MCTS.h"
#include "../src/Inference.cpp"

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
        // Initialize LUTs if necessary (depending on your implementation)
        // HexUtils::init_neighbors();
    }
};

// TEST 1: MCTS vs MCTS (Realistic Game Paths)
// checks consistency after every single move.
TEST_F(ConsistencyTest, EngineSelfPlay_NoCorruption) {
    FastRand rng;
    // Run 5 full games
    for (int game = 0; game < 5; ++game) {
        Position pos(0); // Start Red
        MCTS agent;

        int moves = 0;
        while (pos.getWinner() == -1) {
            // 1. Assert Consistency BEFORE move
            // (If this fails, your program exits, failing the test)
            pos.checkConsistency();

            // 2. Generate Move
            // Use low iterations (e.g. 500) to keep unit tests fast
            // but enough to generate semi-coherent lines.
            int bestMove = agent.search(pos, 5000);
            // 3. ASSERT LEGALITY (New)
            // If this fails, the MCTS selected an occupied or OOB square
            ASSERT_TRUE(pos.isMoveLegal(bestMove))
                << "FATAL: MCTS returned illegal move " << bestMove
                << " at move count " << moves;
            // 3. Make Move
            pos.makeMove(bestMove);
            moves++;

            // 4. Assert Consistency AFTER move
            pos.checkConsistency();

            // Safety break for infinite loops
            if (moves > BOARD_AREA) break;
        }
        std::cout << moves << " moves in game " << game << std::endl;
        // Assert game ended validly
        EXPECT_NE(pos.getWinner(), -1) << "Game " << game << " did not finish.";
    }
}

// TEST 2: Random vs Random (Chaos / Edge Cases)
// Random moves are excellent for finding bitboard overlaps
// because they fill the board in "swiss cheese" patterns.
TEST_F(ConsistencyTest, RandomChaos_NoCorruption) {
    FastRand rng;

    // Run 50 fast random games
    for (int game = 0; game < 50; ++game) {
        Position pos(0);
        int moves = 0;

        while (pos.getWinner() == -1) {
            // Check
            pos.checkConsistency();

            // Pick purely random legal move
            int move = pos.getRandomLegalMove(rng);
            if (move == -1) break; // Should be handled by getWinner/moveCount logic
            // 3. ASSERT LEGALITY (New)
            // If this fails, PDEP or bit-scan logic is broken
            ASSERT_TRUE(pos.isMoveLegal(move))
                << "FATAL: getRandomLegalMove returned occupied square " << move;
            // Move
            pos.makeMove(move);
            moves++;

            // Check
            pos.checkConsistency();
        }
    }
}

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

// TEST: Benchmark MCTS Performance (Nodes Per Second)
TEST_F(PerformanceTest, Calculate_NPS) {
    // 1. Setup
    Position pos(0); // Start with empty board
    MCTS agent;

    // Configuration: 100,000 iterations provides a stable average
    int iterations = 100000;

    std::cout << "[Benchmark] Starting MCTS Search (" << iterations << " iterations)..." << std::endl;

    // 2. Start Timer
    auto start = std::chrono::high_resolution_clock::now();

    // 3. Run Search
    // We discard the return value (best move) as we only care about speed here
    agent.search(pos, iterations);

    // 4. Stop Timer
    auto end = std::chrono::high_resolution_clock::now();

    // 5. Calculate Metrics
    long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Prevent division by zero if it runs instantly
    if (duration_ms == 0) duration_ms = 1;

    double seconds = duration_ms / 1000.0;
    int nps = (int)(iterations / seconds);

    // 6. Report Results
    std::cout << "============================================" << std::endl;
    std::cout << " Total Time: " << duration_ms << " ms" << std::endl;
    std::cout << " Speed:      " << nps << " Nodes/Sec (NPS)" << std::endl;
    std::cout << "============================================" << std::endl;

    // 7. Quality Assertion
    // < 10,000  = Poor (Likely memory allocation issues)
    // 10k - 40k = Good (Standard implementation)
    // > 50,000  = Excellent (Optimized Bitboards/DSU)
    EXPECT_GT(nps, 25000) << "Performance is below the target threshold for this engine.";
}

class OnnxTest : public ::testing::Test {
protected:
    // Helper to print top K moves from policy
    void printTopMoves(const std::vector<float>& policy, int topK = 5) {
        // Create pairs of (index, probability)
        std::vector<std::pair<int, float>> moves;
        for (int i = 0; i < policy.size(); ++i) {
            moves.push_back({i, policy[i]});
        }

        // Sort descending by probability
        std::sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::cout << "--- Top " << topK << " Policy Moves ---" << std::endl;
        for (int i = 0; i < topK && i < moves.size(); ++i) {
            int moveIdx = moves[i].first;
            float prob = moves[i].second;

            // Convert index to A1, B2 notation for readability
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
    // Simple check to ensure model exists (avoids confusing ONNX crash)
    FILE* f = fopen(MODEL_PATH.c_str(), "r");
    if (!f) {
        std::cerr << "[SKIP] Model file not found at: " << MODEL_PATH << std::endl;
        std::cerr << "Please export your PyTorch model to this location to run this test." << std::endl;
        GTEST_SKIP();
    }
    fclose(f);

    std::string wModelPath(MODEL_PATH.begin(), MODEL_PATH.end());
    Inference net(wModelPath);
    Position pos(0);
    FastRand rng;

    // 2. PLAY SOME MOVES (Clutter the board)
    int movesToPlay = 8;
    std::cout << "Playing " << movesToPlay << " random moves..." << std::endl;

    for (int i = 0; i < movesToPlay; ++i) {
        pos.makeRandomRolloutMove(rng);
    }

    // 3. VISUALIZE INPUT
    pos.printPosition();

    // 4. RUN INFERENCE
    std::cout << "Running ONNX Inference..." << std::endl;
    auto result = net.predict(pos);

    std::vector<float> policy = result.first;
    float value = result.second;

    // 5. PRINT RESULTS
    std::cout << "\n================ INFERENCE RESULTS ================" << std::endl;
    std::cout << "Predicted Value (Win Prob for Current Player): " << value << std::endl;

    // Sanity checks
    EXPECT_GE(value, -1.0f);
    EXPECT_LE(value, 1.0f);
    EXPECT_EQ(policy.size(), 121);

    // Check if probabilities sum to roughly 1.0 (if your model outputs softmax)
    // If your model outputs logits, this assertion will fail (which is fine, just remove it).
    float sum = 0.0f;
    for (float p : policy) sum += p;
    std::cout << "Policy Sum: " << sum << std::endl;

    printTopMoves(policy);
    std::cout << "===================================================" << std::endl;
}