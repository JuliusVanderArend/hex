#include <iostream>
#include <limits>
#include "src/Util.h"
#include "src/Position.h"
#include "src/MCTS.h"

#include <chrono>
#include <string>
#include <vector>
#include <sstream>

using namespace engine;

// =============================================================
// GLOBAL STATE
// =============================================================
// We keep these global so the GTP loop can access them easily
std::unique_ptr<Inference> globalNet;
std::unique_ptr<MCTS> globalMCTS;
Position globalPos(0); // 0 = Red/Black (Start), 1 = Blue/White
bool engineRunning = true;

// Configuration
const int SEARCH_ITERATIONS = 30000; // Adjust based on your speed

// =============================================================
// COORDINATE HELPERS
// =============================================================

// Convert "A1", "C10" to index (0-120)
// Returns -1 on invalid
int stringToMove(std::string s) {
    if (s == "swap") return -2; // Handle swap rule if needed
    if (s.length() < 2) return -1;

    // Normalize to lowercase
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    char colChar = s[0];
    std::string rowStr = s.substr(1);

    int col = colChar - 'a';
    int row;
    try {
        row = std::stoi(rowStr) - 1; // 1-based to 0-based
    } catch (...) {
        return -1;
    }

    if (col < 0 || col >= BOARD_SIZE || row < 0 || row >= BOARD_SIZE) return -1;
    return row * BOARD_SIZE + col;
}

// Convert index (0-120) to "A1"
std::string moveToString(int move) {
    if (move < 0) return "resign";

    int row = move / BOARD_SIZE;
    int col = move % BOARD_SIZE;

    std::stringstream ss;
    ss << (char)('a' + col) << (row + 1);
    return ss.str();
}

// =============================================================
// COMMAND HANDLERS
// =============================================================

// Response format: "= response\n\n" (Success) or "? message\n\n" (Failure)
void gtpResponse(std::string msg, bool success = true) {
    if (success) std::cout << "= " << msg << "\n\n";
    else         std::cout << "? " << msg << "\n\n";
    std::cout.flush(); // CRITICAL: Flush output so GUI sees it immediately
}

void cmd_name() { gtpResponse("Group49_HexBot"); }
void cmd_version() { gtpResponse("1.0"); }
void cmd_protocol_version() { gtpResponse("2"); }
void cmd_list_commands() {
    gtpResponse("name\nversion\nprotocol_version\nlist_commands\nboardsize\nclear_board\nplay\ngenmove\nshowboard\nquit");
}

void cmd_boardsize(std::stringstream& ss) {
    int size;
    ss >> size;
    if (size != 11) {
        gtpResponse("unsupported board size (only 11 allowed)", false);
    } else {
        globalPos = Position(0); // Reset
        gtpResponse("");
    }
}

void cmd_clear_board() {
    globalPos = Position(0);
    // Optional: Reset MCTS tree here if your MCTS class supports it
    // globalMCTS = std::make_unique<MCTS>();
    gtpResponse("");
}

void cmd_showboard() {
    // We print the board to stderr so it shows up in logs but doesn't confuse the GUI protocol
    // std::cerr << "\n";
    // globalPos.printPosition();
    gtpResponse("");
}

void cmd_play(std::stringstream& ss) {
    std::string color, coord;
    ss >> color >> coord;

    // ... (Color parsing logic remains the same) ...
    int player = -1;
    char c = std::tolower(color[0]);
    if (c == 'b' || c == 'r') player = 0;
    else if (c == 'w' || c == 'b') player = 1;

    if (player == -1) {
        gtpResponse("invalid color", false);
        return;
    }

    // Check for swap code (-2)
    int move = stringToMove(coord);

    if (move == -2) {
        // Swap is only legal on the second turn (Move Count == 1)
        if (globalPos.moveCount == 1) {
            // Valid Swap.
            // We do NOT modify the board. The 'sideToMove' is already 1 (Blue),
            // and after swap, the 'New Blue' (us) still needs to make a move.
            // So the board state remains identical.
            gtpResponse("");
        } else {
            gtpResponse("illegal move: swap only allowed on turn 2", false);
        }
        return;
    }

    if (move == -1) {
        gtpResponse("invalid coordinate", false);
        return;
    }

    // Standard move handling
    if (player != globalPos.sideToMove) globalPos.sideToMove = player;

    if (!globalPos.isMoveLegal(move)) {
        gtpResponse("illegal move", false);
        return;
    }

    globalPos.makeMove(move);
    gtpResponse("");
}

void cmd_genmove(std::stringstream& ss, InferenceServer& globalServer) {
    std::string color;
    ss >> color;

    // 1. Run MCTS as normal
    SearchResult result = globalMCTS->searchWithPolicy(globalPos, globalServer, SEARCH_ITERATIONS);

    // 2. Check for Swap Opportunity
    std::cerr << "Move Count: " << globalPos.moveCount << std::endl;
    if (globalPos.moveCount == 1) {
        // result.rootValue is the win probability for the current player (Blue).
        // If < 0.5, it means Red (Player 1) has the advantage.
        // We should swap to take the Red position.
        if (result.rootValue < 0.0f) {
            std::cerr << "Eval (" << result.rootValue << ") favors opponent. Swapping." << std::endl;
            gtpResponse("swap");
            globalPos.moveCount++; //will this cause bugs?? because move count no longer equals number of stones on board???
            return;
        }
    }

    if (result.bestMove == -1) {
        gtpResponse("resign");
    } else {
        globalPos.makeMove(result.bestMove);
        gtpResponse(moveToString(result.bestMove));
    }
}

// =============================================================
// MAIN LOOP
// =============================================================

// REPLACE the start of main() in main.cpp with this:
int main(int argc, char* argv[]) {
    // Default fallback
    std::string modelPath = "/home/skynet/git/hex/agents/Group49/models/best.onnx";

    // Accept model path from Arbiter/Command Line
    if (argc > 1) {
        modelPath = argv[1];
    }
    Inference net(modelPath); // Use the variable, not the constant
    InferenceServer globalServer(net);
    try {
        // If your NeuralNet constructor takes wstring (Windows), convert it.
        // Since we are on Linux now, standard string is usually fine if you updated NeuralNet.h
        // If not, use: std::wstring wModelPath(modelFile.begin(), modelFile.end());
        // globalNet = std::make_unique<NeuralNet>(wModelPath);

        // Assuming Linux-friendly NeuralNet:
        // globalNet = std::make_unique<NeuralNet>(modelFile);

        // For now, if MCTS creates its own net or doesn't need it passed explicitly:
        globalMCTS = std::make_unique<MCTS>();

        // std::cerr << "Engine initialized. Model: " << modelFile << std::endl;

    } catch (const std::exception& e) {
        // std::cerr << "Failed to load model: " << e.what() << std::endl;
        return 1;
    }

    // 2. Command Loop
    std::string line;
    while (engineRunning && std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string command;
        ss >> command;

        if (command == "name") cmd_name();
        else if (command == "version") cmd_version();
        else if (command == "protocol_version") cmd_protocol_version();
        else if (command == "list_commands") cmd_list_commands();
        else if (command == "boardsize") cmd_boardsize(ss);
        else if (command == "clear_board") cmd_clear_board();
        else if (command == "showboard") cmd_showboard();
        else if (command == "play") cmd_play(ss);
        else if (command == "genmove") cmd_genmove(ss,globalServer);
        else if (command == "quit") {
            engineRunning = false;
            gtpResponse("");
        }
        else {
            gtpResponse("unknown command", false);
        }
    }

    return 0;
}
