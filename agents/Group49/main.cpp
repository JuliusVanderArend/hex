#include <iostream>
#include <limits>
#include "src/Util.h"
#include "src/Position.h"
#include "src/MCTS.h"
#include <chrono>
#include <string>
#include <vector>
#include <sstream>
#include <windows.h> // Required for spawning the MoHex process

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

// int main() {
//     // Disable I/O buffering so the GUI sees responses instantly
//     std::cout.setf(std::ios::unitbuf);
//
//     // Engine State
//     Position pos(0);
//     MCTS agent;
//
//     std::string line;
//     while (std::getline(std::cin, line)) {
//         if (line.empty()) continue;
//
//         std::vector<std::string> tokens = split(line);
//         if (tokens.empty()) continue;
//
//         std::string command = tokens[0];
//         std::string cmd_id = "";
//
//         // GTP commands might have an ID number first (e.g., "10 genmove black")
//         // We need to parse and preserve it for the response.
//         if (isdigit(command[0])) {
//             cmd_id = command;
//             if (tokens.size() > 1) command = tokens[1];
//             // Shift tokens if needed, or just look at specific args
//         }
//
//         // --- COMMAND HANDLING ---
//
//         if (command == "name") {
//             std::cout << "=" << cmd_id << " MyHexBot" << std::endl;
//         }
//         else if (command == "protocol_version") {
//             std::cout << "=" << cmd_id << " 2" << std::endl;
//         }
//         else if (command == "version") {
//             std::cout << "=" << cmd_id << " 1.0" << std::endl;
//         }
//         else if (command == "boardsize") {
//             // MoHex sends "boardsize 11"
//             // We assume 11, but you could assert check here
//             std::cout << "=" << cmd_id << std::endl;
//         }
//         else if (command == "clear_board") {
//             pos = Position(0); // Reset
//             std::cout << "=" << cmd_id << std::endl;
//         }
//         else if (command == "play") {
//             // usage: play black c5
//             // tokens: [play, black, c5] OR [ID, play, black, c5]
//
//             std::string moveStr = tokens.back(); // The move is usually last
//             int moveIdx = stringToIndex(moveStr);
//
//             // Check legality (optional but recommended)
//             pos.makeMove(moveIdx);
//
//             std::cout << "=" << cmd_id << std::endl;
//         }
//         else if (command == "genmove") {
//             // usage: genmove black
//             // The GUI is asking US to move.
//
//             // 1. Run MCTS
//             // Use time-based search for real play (e.g., 5 seconds)
//             // Or fixed iterations
//             int bestMove = agent.search(pos, 1000000);
//
//             // 2. Apply it internally
//             pos.makeMove(bestMove);
//
//             // 3. Respond
//             std::cout << "=" << cmd_id << " " << indexToString(bestMove) << std::endl;
//         }
//         else if (command == "quit") {
//             std::cout << "=" << cmd_id << std::endl;
//             break;
//         }
//         else if (command == "known_command") {
//             // "known_command genmove" -> "true"
//             std::cout << "=" << cmd_id << " true" << std::endl;
//         }
//         else if (command == "list_commands") {
//             std::cout << "=" << cmd_id << "\nname\nversion\nboardsize\nclear_board\nplay\ngenmove\nquit\n" << std::endl;
//         }
//         else {
//             // Unknown command
//             std::cout << "?" << cmd_id << " unknown command" << std::endl;
//         }
//
//         // Critical: GTP requires a double newline after response
//         std::cout << std::endl;
//     }
//
//     return 0;
// }

const std::string MOHEX_PATH = "C:\\Users\\gamin\\Desktop\\mohex_wsl.bat";

// --- WINDOWS PIPE WRAPPER ---
// This class handles the complex task of talking to the .bat file
class MoHexConnection {
private:
    HANDLE hChildStd_IN_Rd = NULL;
    HANDLE hChildStd_IN_Wr = NULL;
    HANDLE hChildStd_OUT_Rd = NULL;
    HANDLE hChildStd_OUT_Wr = NULL;

public:
    MoHexConnection() {
        SECURITY_ATTRIBUTES saAttr;
        saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
        saAttr.bInheritHandle = TRUE;
        saAttr.lpSecurityDescriptor = NULL;

        // Create a pipe for the child process's STDOUT.
        CreatePipe(&hChildStd_OUT_Rd, &hChildStd_OUT_Wr, &saAttr, 0);
        SetHandleInformation(hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0);

        // Create a pipe for the child process's STDIN.
        CreatePipe(&hChildStd_IN_Rd, &hChildStd_IN_Wr, &saAttr, 0);
        SetHandleInformation(hChildStd_IN_Wr, HANDLE_FLAG_INHERIT, 0);

        // Create the child process.
        STARTUPINFOA siStartInfo;
        PROCESS_INFORMATION piProcInfo;
        ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));
        ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
        siStartInfo.cb = sizeof(STARTUPINFO);
        siStartInfo.hStdError = hChildStd_OUT_Wr; // Redirect stderr to same pipe
        siStartInfo.hStdOutput = hChildStd_OUT_Wr;
        siStartInfo.hStdInput = hChildStd_IN_Rd;
        siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

        // Command line must be mutable for CreateProcess
        std::string cmd = "cmd.exe /c \"" + MOHEX_PATH + "\"";
        char* cmdCstr = &cmd[0];

        BOOL success = CreateProcessA(NULL, cmdCstr, NULL, NULL, TRUE, 0, NULL, NULL, &siStartInfo, &piProcInfo);

        if (!success) {
            std::cerr << "Failed to start MoHex! Error: " << GetLastError() << std::endl;
            exit(1);
        }

        // Close handles to the pipe endpoints we don't need
        CloseHandle(hChildStd_OUT_Wr);
        CloseHandle(hChildStd_IN_Rd);
    }

    // Send a command to MoHex
    void send(std::string cmd) {
        cmd += "\n"; // GTP requires newline
        DWORD dwWritten;
        WriteFile(hChildStd_IN_Wr, cmd.c_str(), cmd.length(), &dwWritten, NULL);
    }

    // Read response until we find the empty line (GTP standard)
    std::string readResponse() {
        DWORD dwRead;
        CHAR chBuf[4096];
        std::string output = "";
        bool responseComplete = false;

        // Keep reading until we see the double newline standard in GTP
        // Or simpler: usually MoHex replies with "= [result]\n\n"
        while (true) {
            BOOL bSuccess = ReadFile(hChildStd_OUT_Rd, chBuf, 4095, &dwRead, NULL);
            if (!bSuccess || dwRead == 0) break;

            chBuf[dwRead] = '\0';
            output += chBuf;

            // GTP responses end with a double newline
            if (output.find("\n\n") != std::string::npos) break;
        }

        // Clean up output (remove "= " and newlines)
        size_t equalSign = output.find("=");
        if (equalSign != std::string::npos) {
            output = output.substr(equalSign + 1);
        }

        // Trim whitespace
        const std::string whitespace = " \n\r\t";
        size_t first = output.find_first_not_of(whitespace);
        if (std::string::npos == first) return "";
        size_t last = output.find_last_not_of(whitespace);
        return output.substr(first, (last - first + 1));
    }
};

// --- MAIN GAME LOOP ---

int main() {
    // 1. Initialize Internal Agent
    Position pos(0);
    MCTS agent;

    // 2. Initialize External MoHex
    std::cout << "Starting MoHex from: " << MOHEX_PATH << "..." << std::endl;
    MoHexConnection mohex;

    // Handshake with MoHex
    mohex.send("name");
    std::cout << "Connected to: " << mohex.readResponse() << std::endl;

    mohex.send("boardsize " + std::to_string(BOARD_SIZE));
    mohex.readResponse(); // Consume response

    mohex.send("uct_param_search max_games 1");
    mohex.readResponse();

    mohex.send("uct_param_search");
    std::cout << "Search Params:\n" << mohex.readResponse() << std::endl;

    mohex.send("uct_param_search time_limit 0.1");

    mohex.readResponse(); // Consume the "=" response
    // List all player parameters
    mohex.send("uct_param_player");
    std::cout << "Player Params:\n" << mohex.readResponse() << std::endl;

    std::cout << "Starting match: Internal MCTS (Black) vs MoHex (White)" << std::endl;

    // GAME LOOP
    int moveCount = 0;
    while (true) {
        pos.checkConsistency();
        pos.printPosition();
        // --- BLACK (Internal MCTS) ---
        std::cout << "\nThinking..." << std::endl;
        int myMove = agent.search(pos, 500000); // 500k iterations
        std::string myMoveStr = indexToString(myMove); // e.g., "C5"

        std::cout << "MCTS (Black) plays: " << myMoveStr << std::endl;

        // 1. Update internal board
        pos.makeMove(myMove);
        if (pos.getWinner() != -1) {
            std::cout << "Game Over! " << pos.getWinner() << " wins!" << std::endl;
            break;
        }

        // 2. Tell MoHex what we did
        mohex.send("play black " + myMoveStr);
        std::string response = mohex.readResponse();
        if (response.find("illegal") != std::string::npos) {
            std::cerr << "CRITICAL: MoHex rejected our move!" << std::endl;
            break;
        }

        // Check Win/Loss (Check your Position class logic here)
        // if (pos.checkWin()) ...

        // --- WHITE (External MoHex) ---
        std::cout << "MoHex is thinking..." << std::endl;
        // 1. Ask MoHex for its move
        mohex.send("genmove white");
        std::string mohexMoveStr = mohex.readResponse();

        // DEBUG: Print exactly what we got (in quotes to see empty strings)
        std::cout << "MoHex (White) raw response: '" << mohexMoveStr << "'" << std::endl;

        // CHECK 1: Handle Empty/Crash
        if (mohexMoveStr.empty()) {
            std::cerr << "CRITICAL ERROR: MoHex returned an empty response. It likely crashed." << std::endl;
            break; // Exit the game loop
        }

        // CHECK 2: Handle Resignation
        // MoHex sends "resign" when it knows it has lost.
        if (mohexMoveStr.find("resign") != std::string::npos) {
            std::cout << "MoHex resigns! You win!" << std::endl;
            break;
        }

        std::cout << "MoHex (White) plays: " << mohexMoveStr << std::endl;

        // 2. Update internal board
        int mohexMove = stringToIndex(mohexMoveStr);
        pos.makeMove(mohexMove);
        if (pos.getWinner() != -1) {
            std::cout << "Game Over! " << pos.getWinner() << " wins!" << std::endl;
            break;
        }

        // Check Win/Loss
        moveCount++;
        if (moveCount > (BOARD_SIZE * BOARD_SIZE)) break; // Safety break
    }

    mohex.send("quit");
    return 0;
}