#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#include "src/MCTS.h"
#include "src/Position.h"
#include "src/Util.h"

using namespace engine;

// Mutex for thread-safe file writing and console output
std::mutex io_mutex;

// Updated Sample struct to hold all 6 planes explicitly
// This makes it easy to write clear JSON keys
struct Sample {
    int playerToMove;
    std::array<uint8_t, BOARD_AREA> red;         // Plane 0
    std::array<uint8_t, BOARD_AREA> blue;        // Plane 1
    std::array<uint8_t, BOARD_AREA> turn;        // Plane 2
    std::array<uint8_t, BOARD_AREA> last_move;   // Plane 3
    std::array<uint8_t, BOARD_AREA> conn_start;  // Plane 4
    std::array<uint8_t, BOARD_AREA> conn_end;    // Plane 5

    std::array<double, BOARD_AREA> policy;
};

// Helper: Extract a specific channel from the flat float vector
std::array<uint8_t, BOARD_AREA> extractPlane(const std::vector<float>& tensor, int channelIdx) {
    std::array<uint8_t, BOARD_AREA> plane{};
    int offset = channelIdx * BOARD_AREA;
    for (int i = 0; i < BOARD_AREA; ++i) {
        // Safe to cast 1.0f/0.0f to uint8
        plane[i] = static_cast<uint8_t>(tensor[offset + i]);
    }
    return plane;
}

std::string planeToJson(const std::array<uint8_t, BOARD_AREA>& plane) {
    std::ostringstream oss;
    oss << '[';
    for (int i = 0; i < BOARD_AREA; ++i) {
        if (i != 0) oss << ',';
        oss << static_cast<int>(plane[i]);
    }
    oss << ']';
    return oss.str();
}

std::string policyToJson(const std::array<double, BOARD_AREA>& policy) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << '[';
    oss << std::setprecision(6);
    for (int i = 0; i < BOARD_AREA; ++i) {
        if (i != 0) oss << ',';
        oss << policy[i];
    }
    oss << ']';
    return oss.str();
}

struct GameSamples {
    std::vector<Sample> samples;
    int winner;
};

GameSamples playSelfPlayGame(MCTS& agent, int iterations) {
    Position pos(0);
    GameSamples record;

    while (pos.getWinner() == -1) {
        // Run MCTS
        SearchResult result = agent.searchWithPolicy(pos, iterations);

        // Fallback for no moves
        if (result.bestMove < 0) {
            auto legal = pos.getLegalMoves();
            if (legal.empty()) break;
            result.bestMove = legal.front();
        }

        Sample sample;
        sample.playerToMove = pos.sideToMove;

        // --- NEW LOGIC: Extract All 6 Planes ---
        // Requires Position::toTensor() to produce 6 channels:
        // 0:Red, 1:Blue, 2:Turn, 3:LastMove, 4:Start, 5:End
        std::vector<float> tensor = pos.toTensor();

        sample.red        = extractPlane(tensor, 0);
        sample.blue       = extractPlane(tensor, 1);
        sample.turn       = extractPlane(tensor, 2);
        sample.last_move  = extractPlane(tensor, 3); // Check index match with Position.cpp
        sample.conn_start = extractPlane(tensor, 4);
        sample.conn_end   = extractPlane(tensor, 5);

        sample.policy = result.policy;
        record.samples.push_back(sample);

        pos.makeMove(result.bestMove);
    }

    record.winner = pos.getWinner();
    if (record.winner == -1) record.winner = 2; // Draw

    return record;
}

void worker(int iterations, int totalGames, std::atomic<int>& gamesPlayed, std::ofstream& out) {
    // Thread-local MCTS agent (critical for thread safety)
    MCTS agent;

    while (true) {
        // Fetch next game index
        int gameIdx = gamesPlayed.fetch_add(1);
        if (gameIdx >= totalGames) return;

        // Play
        GameSamples record = playSelfPlayGame(agent, iterations);

        // Save Result (Locking)
        std::lock_guard<std::mutex> lock(io_mutex);

        for (size_t moveIdx = 0; moveIdx < record.samples.size(); ++moveIdx) {
            const Sample& sample = record.samples[moveIdx];
            int value = 0;
            if (record.winner != 2) {
                value = (record.winner == sample.playerToMove) ? 1 : -1;
            }

            // Write all 6 planes to JSON
            out << '{'
                << "\"game\":" << gameIdx
                << ",\"move\":" << moveIdx
                << ",\"player\":" << sample.playerToMove
                << ",\"value\":" << value
                << ",\"red\":" << planeToJson(sample.red)
                << ",\"blue\":" << planeToJson(sample.blue)
                << ",\"turn\":" << planeToJson(sample.turn)
                << ",\"last_move\":" << planeToJson(sample.last_move)
                << ",\"conn_start\":" << planeToJson(sample.conn_start)
                << ",\"conn_end\":" << planeToJson(sample.conn_end)
                << ",\"policy\":" << policyToJson(sample.policy)
                << "}\n";
        }

        // Optional: Progress log
        if ((gameIdx + 1) % 1 == 0) {
            std::cout << "Finished game " << gameIdx + 1 << "/" << totalGames
                      << " (" << record.samples.size() << " moves) [Thread "
                      << std::this_thread::get_id() << "]" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    int games = 1;
    int iterations = 10000;
    std::string outputPath = "selfplay_data.jsonl";

    if (argc > 1) games = std::stoi(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) outputPath = argv[3];

    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    // Auto-detect threads
    unsigned int nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0) nThreads = 1;

    std::cout << "Starting Parallel Self-Play" << std::endl;
    std::cout << "Games: " << games << " | Iterations: " << iterations << " | Threads: " << nThreads << std::endl;

    std::vector<std::thread> threads;
    std::atomic<int> gamesPlayed{0};

    for (unsigned int i = 0; i < nThreads; ++i) {
        threads.emplace_back(worker, iterations, games, std::ref(gamesPlayed), std::ref(out));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    std::cout << "All games completed. Saved to " << outputPath << std::endl;
    return 0;
}