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

// int main() {
//     FastRand rng;
//     Position pos(0);
//
//     long long totalMoves = 0;
//     int redWins = 0;
//     int blueWins = 0;
//     int nowin = 0;
//
//     auto start = std::chrono::high_resolution_clock::now();
//
//     int iterations = 1000000;
//
//     for (int i = 0; i < iterations; ++i) {
//         Position workingPos = pos; // Copy
//
//         // Loop until game ends
//         while (workingPos.getWinner() == -1) {
//             workingPos.makeRandomRolloutMove(rng);
//         }
//
//         if (workingPos.getWinner() == 0) redWins++;
//         else if (workingPos.getWinner() == 1) blueWins++;
//         else nowin++;
//
//         totalMoves += workingPos.moveCount;
//     }
//
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//
//     std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
//     std::cout << "Games/Sec:  " << (iterations * 1000LL) / duration.count() << std::endl;
//
//     // DIAGNOSTICS
//     double avgMoves = (double)totalMoves / iterations;
//     std::cout << "Average Moves per Game: " << avgMoves << std::endl;
//     std::cout << "Red Win Rate: " << (double)redWins / iterations << std::endl;
//
//     std::cout << "nowin Rate: " << (double)nowin / iterations << std::endl;
//
//
//     return 0;
// }
// int main() {
//     // 1. Setup
//     // Initialize the DSU/Neighbor LUTs (if you used the struct Initializer approach)
//     // HexUtils::init_neighbors(); // Uncomment if your LUT isn't constexpr
//
//     // Create the game state (Red starts)
//     Position pos(0);
//
//     // Create the AI
//     MCTS agent;
//
//     // Set AI strength (Iterations per move)
//     // 50,000 is usually a strong, quick move (~0.5 - 1.0 second)
//     // 200,000+ is very strong but slower.
//     const int AI_ITERATIONS = 500000;
//
//     std::cout << "======================================" << std::endl;
//     std::cout << "      HEX - HUMAN (Red) vs AI (Blue)  " << std::endl;
//     std::cout << "======================================" << std::endl;
//     std::cout << "Instructions:" << std::endl;
//     std::cout << " - Enter moves as a single integer (0 - 120)." << std::endl;
//     std::cout << " - 0 is Top-Left (A1). 10 is Top-Right (K1)." << std::endl;
//     std::cout << " - 11 is Row 2, Col A (A2), etc." << std::endl;
//     std::cout << "======================================" << std::endl;
//
//     pos.printPosition();
//
//     // 2. Game Loop
//     while (pos.getWinner() == -1) {
//
//         if (pos.sideToMove == 0) {
//             // --- HUMAN TURN (RED) ---
//             int move = -1;
//             bool valid = false;
//
//             while (!valid) {
//                 std::cout << "\n[Red/Human] Enter move (0-" << (BOARD_AREA - 1) << "): ";
//
//                 if (!(std::cin >> move)) {
//                     // Handle non-integer input
//                     std::cin.clear();
//                     std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
//                     std::cout << "Invalid input. Please enter a number." << std::endl;
//                     continue;
//                 }
//
//                 // Validation 1: Range
//                 if (move < 0 || move >= BOARD_AREA) {
//                     std::cout << "Out of bounds! Choose between 0 and 120." << std::endl;
//                     continue;
//                 }
//
//                 // Validation 2: Occupancy
//                 // We need to check if the bit is already set in 'occupancy'
//                 // Note: occupancy is private, but you likely have a helper or can check via legal moves
//                 // Assuming Position::getLegalMoves() exists or we trust the bitboard logic
//                 // A quick dirty check if you don't have isOccupied():
//                 std::vector<int> legal = pos.getLegalMoves();
//                 bool found = false;
//                 for(int m : legal) { if(m == move) found = true; }
//
//                 if (!found) {
//                     std::cout << "Square already occupied! Pick another." << std::endl;
//                 } else {
//                     valid = true;
//                 }
//             }
//
//             pos.makeMove(move);
//         }
//         else {
//             // --- AI TURN (BLUE) ---
//             std::cout << "\n[Blue/AI] Thinking (" << AI_ITERATIONS << " iter)..." << std::flush;
//
//             auto start = std::chrono::high_resolution_clock::now();
//
//             // RUN MCTS
//             int bestMove = agent.search(pos, AI_ITERATIONS);
//
//             auto end = std::chrono::high_resolution_clock::now();
//             auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//
//             std::cout << " Done! (" << ms << "ms)" << std::endl;
//             std::cout << "[Blue/AI] Plays: " << bestMove << std::endl;
//
//             pos.makeMove(bestMove);
//         }
//
//         // Display board after every move
//         pos.printPosition();
//     }
//
//     // 3. Game Over
//     int winner = pos.getWinner();
//     std::cout << "\n======================================" << std::endl;
//     if (winner == 0) {
//         std::cout << "           RED (HUMAN) WINS!          " << std::endl;
//     } else {
//         std::cout << "           BLUE (AI) WINS!            " << std::endl;
//     }
//     std::cout << "======================================" << std::endl;
//
//     return 0;
// }

// =============================================================
// GLOBAL STATE
// =============================================================
// We keep these global so the GTP loop can access them easily
std::unique_ptr<Inference> globalNet;
std::unique_ptr<MCTS> globalMCTS;
Position globalPos(0); // 0 = Red/Black (Start), 1 = Blue/White
Inference net(MODEL_PATH);
InferenceServer globalServer(net);
bool engineRunning = true;

// Configuration
const int SEARCH_ITERATIONS = 10000; // Adjust based on your speed

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

    // Parse Color (w/white or b/black)
    // HexGui often sends "play white A1"
    int player = -1;
    char c = std::tolower(color[0]);
    if (c == 'b' || c == 'r') player = 0; // Black/Red
    else if (c == 'w' || c == 'b') player = 1; // White/Blue

    if (player == -1) {
        gtpResponse("invalid color", false);
        return;
    }

    // Check if move matches current turn (HexGui might try to force out-of-turn play)
    if (player != globalPos.sideToMove) {
        // In some protocols, we might accept this by forcing the side to move
        globalPos.sideToMove = player;
    }

    int move = stringToMove(coord);
    if (move == -1) {
        gtpResponse("invalid coordinate", false);
        return;
    }

    if (move == -2) {
        // Swap logic (if supported)
        gtpResponse("swap not implemented", false);
        return;
    }

    globalPos.makeMove(move);
    gtpResponse("");
}

void cmd_genmove(std::stringstream& ss) {
    std::string color;
    ss >> color; // e.g. "genmove white"

    // Optional: Ensure side to move matches requested color
    // ...

    // RUN MCTS
    // Note: If you implemented the InferenceServer, pass it here or ensure globalMCTS uses it.
    // Assuming standard MCTS usage:

    // Log for debugging
    std::cerr << "Thinking..." << std::endl;

    // Call your MCTS search
    // If you need to pass the NeuralNet, make sure your MCTS class accepts it
    // e.g. globalMCTS->search(globalPos, *globalNet, SEARCH_ITERATIONS);

    // Based on your self_play.cpp:
    SearchResult result = globalMCTS->searchWithPolicy(globalPos,globalServer, SEARCH_ITERATIONS);

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

int main(int argc, char* argv[]) {
    // 1. Load Model (Optional: From argv)
    std::string modelFile = MODEL_PATH;
    if (argc > 1) modelFile = argv[1];

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
        else if (command == "genmove") cmd_genmove(ss);
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
