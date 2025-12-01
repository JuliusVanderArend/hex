#include <iostream>
#include "src/Util.h"
#include "src/Position.h"
#include <chrono>

using namespace engine;

int main() {
    FastRand rng;
    Position pos(0); // Start empty

    long long totalMoves = 0;
    int redWins = 0;
    int blueWins = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Run 10 million games
    int iterations = 1000000;

    for (int i = 0; i < iterations; ++i) {
        Position workingPos = pos; // Copy

        // Loop until game ends
        while (workingPos.getWinner() == -1) {
            workingPos.makeRandomRolloutMove(rng); // Pass RNG if removed from class
        }

        // 1. ACCUMULATE STATS (Prevents Dead Code Elimination)
        if (workingPos.getWinner() == 0) redWins++;
        else blueWins++;

        // 2. TRACK MOVES (Detects Instant-Win Bug)
        totalMoves += workingPos.moveCount;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Games/Sec:  " << (iterations * 1000LL) / duration.count() << std::endl;

    // CRITICAL DIAGNOSTICS
    double avgMoves = (double)totalMoves / iterations;
    std::cout << "Average Moves per Game: " << avgMoves << std::endl;
    std::cout << "Red Win Rate: " << (double)redWins / iterations << std::endl;

    // Sanity Check Assertions
    if (avgMoves < 20.0) {
        std::cout << "WARNING: Games are ending too fast! Check isWon() logic." << std::endl;
    }

    return 0;
}