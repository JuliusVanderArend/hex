#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include "Inference.h"
#include "Position.h"
#include "Util.h" // For BOARD_SIZE

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