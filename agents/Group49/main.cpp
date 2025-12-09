#include <iostream>
#include <limits>
#include "src/Util.h"
#include "src/Position.h"
#include "src/MCTS.h"
#include <chrono>

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

std::vector<std::string> split(const std::string &str) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (ss >> token) tokens.push_back(token);
    return tokens;
}

int main() {
    // Disable I/O buffering so the GUI sees responses instantly
    std::cout.setf(std::ios::unitbuf);

    // Engine State
    Position pos(0);
    MCTS agent;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::vector<std::string> tokens = split(line);
        if (tokens.empty()) continue;

        std::string command = tokens[0];
        std::string cmd_id = "";

        // GTP commands might have an ID number first (e.g., "10 genmove black")
        // We need to parse and preserve it for the response.
        if (isdigit(command[0])) {
            cmd_id = command;
            if (tokens.size() > 1) command = tokens[1];
            // Shift tokens if needed, or just look at specific args
        }

        // --- COMMAND HANDLING ---

        if (command == "name") {
            std::cout << "=" << cmd_id << " MyHexBot" << std::endl;
        }
        else if (command == "protocol_version") {
            std::cout << "=" << cmd_id << " 2" << std::endl;
        }
        else if (command == "version") {
            std::cout << "=" << cmd_id << " 1.0" << std::endl;
        }
        else if (command == "boardsize") {
            // MoHex sends "boardsize 11"
            // We assume 11, but you could assert check here
            std::cout << "=" << cmd_id << std::endl;
        }
        else if (command == "clear_board") {
            pos = Position(0); // Reset
            std::cout << "=" << cmd_id << std::endl;
        }
        else if (command == "play") {
            // usage: play black c5
            // tokens: [play, black, c5] OR [ID, play, black, c5]

            std::string moveStr = tokens.back(); // The move is usually last
            int moveIdx = stringToIndex(moveStr);

            // Check legality (optional but recommended)
            pos.makeMove(moveIdx);

            std::cout << "=" << cmd_id << std::endl;
        }
        else if (command == "genmove") {
            // usage: genmove black
            // The GUI is asking US to move.

            // 1. Run MCTS
            // Use time-based search for real play (e.g., 5 seconds)
            // Or fixed iterations
            int bestMove = agent.search(pos, 1000000);

            // 2. Apply it internally
            pos.makeMove(bestMove);

            // 3. Respond
            std::cout << "=" << cmd_id << " " << indexToString(bestMove) << std::endl;
        }
        else if (command == "quit") {
            std::cout << "=" << cmd_id << std::endl;
            break;
        }
        else if (command == "known_command") {
            // "known_command genmove" -> "true"
            std::cout << "=" << cmd_id << " true" << std::endl;
        }
        else if (command == "list_commands") {
            std::cout << "=" << cmd_id << "\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit\n" << std::endl;
        }
        else {
            // Unknown command
            std::cout << "?" << cmd_id << " unknown command" << std::endl;
        }

        // Critical: GTP requires a double newline after response
        std::cout << std::endl;
    }

    return 0;
}