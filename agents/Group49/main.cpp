#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include "src/Util.h"
#include "src/Position.h"
#include "src/MCTS.h"

using namespace std;
using namespace engine;

static constexpr int SEARCH_ITERATIONS = 100000;

vector<string> split(const string& s, char delim) {
    vector<string> elems;
    string item;
    stringstream ss(s);
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 3) return 1;

    string myColour = argv[1];

    Inference net("agents/Group49/models/best256");
    InferenceServer server(net);
    MCTS mcts;

    Position pos(myColour == "R" ? 0 : 1);

    const int OPENING_X = 0;
    const int OPENING_Y = 1;
    const float SWAP_THRESHOLD = -0.05f;

    string line;
    while (getline(cin, line)) {
        if (line.empty()) continue;

        auto parts = split(line, ';');
        if (parts.size() < 4) continue;

        string command = parts[0];
        string moveStr = parts[1];
        int turn = stoi(parts[3]);

        if (command == "START") {
            pos = Position(myColour == "R" ? 0 : 1);
        }
        else if (command == "CHANGE") {
            int x, y;
            sscanf(moveStr.c_str(), "%d,%d", &x, &y);
            pos.makeMove(y * BOARD_SIZE + x);
        }
        else if (command == "SWAP") {
            pos.sideToMove ^= 1;
        }

        int bestMove = -1;
        bool doSwap = false;

        if (pos.moveCount == 0) {
            bestMove = OPENING_Y * BOARD_SIZE + OPENING_X;
        }
        else if (pos.moveCount == 1) {
            SearchResult result = mcts.searchWithPolicy(pos, server, SEARCH_ITERATIONS);

            if (result.rootValue < SWAP_THRESHOLD) {
                doSwap = true;
            } else {
                bestMove = result.bestMove;
            }
        }
        else {
            SearchResult result = mcts.searchWithPolicy(pos, server, SEARCH_ITERATIONS);
            bestMove = result.bestMove;
        }

        if (doSwap) {
            cout << "SWAP" << "\n";
            cout.flush();
            pos.sideToMove ^= 1;
        } else {
            pos.makeMove(bestMove);
            int x = bestMove % BOARD_SIZE;
            int y = bestMove / BOARD_SIZE;
            cout << x << "," << y << "\n";
            cout.flush();
        }
    }

    return 0;
}