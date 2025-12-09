#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/MCTS.h"
#include "src/Position.h"
#include "src/Util.h"

using namespace engine;

struct Sample {
    int playerToMove;
    std::array<uint8_t, BOARD_AREA> red;
    std::array<uint8_t, BOARD_AREA> blue;
    std::array<double, BOARD_AREA> policy;
};

std::array<uint8_t, BOARD_AREA> buildPlane(const Position& pos, int player) {
    std::array<uint8_t, BOARD_AREA> plane{};
    for (int idx = 0; idx < BOARD_AREA; ++idx) {
        plane[idx] = pos.hasStone(player, idx) ? 1 : 0;
    }
    return plane;
}

std::string planeToString(const std::array<uint8_t, BOARD_AREA>& plane) {
    std::string serialized;
    serialized.reserve(BOARD_AREA);
    for (uint8_t value : plane) {
        serialized.push_back(value ? '1' : '0');
    }
    return serialized;
}

std::string policyToString(const std::array<double, BOARD_AREA>& policy) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(6);
    for (int i = 0; i < BOARD_AREA; ++i) {
        if (i != 0) oss << ' ';
        oss << policy[i];
    }
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
        SearchResult result = agent.searchWithPolicy(pos, iterations);
        if (result.bestMove < 0) {
            auto legal = pos.getLegalMoves();
            if (legal.empty()) {
                break;
            }
            result.bestMove = legal.front();
        }

        Sample sample;
        sample.playerToMove = pos.sideToMove;
        sample.red = buildPlane(pos, 0);
        sample.blue = buildPlane(pos, 1);
        sample.policy = result.policy;
        record.samples.push_back(sample);

        pos.makeMove(result.bestMove);
    }

    record.winner = pos.getWinner();
    if (record.winner == -1) {
        record.winner = 2; // treat unresolved as draw
    }
    return record;
}

int main(int argc, char** argv) {
    int games = 1;
    int iterations = 10000;
    std::string outputPath = "selfplay_data.csv";

    if (argc > 1) games = std::stoi(argv[1]);
    if (argc > 2) iterations = std::stoi(argv[2]);
    if (argc > 3) outputPath = argv[3];

    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    out << "game,move,player,value,red,blue,policy\n";

    MCTS agent;
    for (int gameIdx = 0; gameIdx < games; ++gameIdx) {
        GameSamples record = playSelfPlayGame(agent, iterations);

        for (size_t moveIdx = 0; moveIdx < record.samples.size(); ++moveIdx) {
            const Sample& sample = record.samples[moveIdx];
            int value = 0;
            if (record.winner != 2) {
                value = (record.winner == sample.playerToMove) ? 1 : -1;
            }

            out << gameIdx << ',' << moveIdx << ',' << sample.playerToMove << ',' << value
                << ",\"" << planeToString(sample.red) << "\",\"" << planeToString(sample.blue)
                << "\",\"" << policyToString(sample.policy) << "\"\n";
        }

        std::cout << "Finished game " << gameIdx + 1 << "/" << games
                  << " (" << record.samples.size() << " moves)" << std::endl;
    }

    return 0;
}
