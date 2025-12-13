#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <cstring>
#include <algorithm>
#include <map>

#include "src/Position.h"
#include "src/Util.h"

using namespace engine;

// --- CONFIGURATION ---
const int OPENING_PLIES = 8;
const int SAFETY_LIMIT = 200;

std::mutex print_mutex;

struct GameResult {
    int winner;
    int blackEngineID;
    std::string finalString;
    bool crashed = false;
};

struct Stats {
    int winsA = 0;
    int winsB = 0;
    int draws = 0;
    int crashes = 0;
    int gamesPlayed = 0;
};

// --- HELPER FUNCTIONS ---
void saveGameToSGF(const std::string& filename,
                   const std::string& blackName,
                   const std::string& whiteName,
                   int winner, // 0=Black, 1=White
                   const std::vector<std::string>& moves)
{
    std::ofstream file(filename);
    if (!file.is_open()) return;

    // SGF Header
    // GM[11] = Hex, SZ[11] = Size 11x11
    file << "(;FF[4]GM[11]SZ[11]\n";
    file << "PB[" << blackName << "]PW[" << whiteName << "]\n";

    // Result
    if (winner == 0) file << "RE[B+Resign]\n";
    else if (winner == 1) file << "RE[W+Resign]\n";
    else file << "RE[Draw]\n";

    file << "DT[" << __DATE__ << "]\n";

    // Write Moves
    for (size_t i = 0; i < moves.size(); ++i) {
        // Even indices (0, 2...) are Black, Odd (1, 3...) are White
        char player = (i % 2 == 0) ? 'B' : 'W';
        file << ";" << player << "[" << moves[i] << "]\n";
    }

    file << ")\n";
    file.close();
}

std::vector<std::string> splitCommand(const std::string& cmd) {
    std::istringstream iss(cmd);
    std::vector<std::string> args;
    std::string arg;
    while (iss >> arg) args.push_back(arg);
    return args;
}

// Extracts "Group49" from "./cmake-build/Group49"
std::string getBaseName(const std::string& path) {
    std::string clean = path;
    // Remove args first
    size_t space = clean.find(' ');
    if (space != std::string::npos) clean = clean.substr(0, space);

    // Find last slash
    size_t lastSlash = clean.find_last_of("/\\");
    if (lastSlash != std::string::npos) clean = clean.substr(lastSlash + 1);

    return clean;
}

std::string moveToString(int move) {
    if (move < 0) return "resign";
    int row = move / BOARD_SIZE;
    int col = move % BOARD_SIZE;
    std::stringstream ss;
    ss << (char)('a' + col) << (row + 1);
    return ss.str();
}

int stringToMove(std::string s) {
    if (s.length() < 2) return -1;
    char colChar = std::tolower(s[0]);
    std::string rowStr = s.substr(1);
    int col = colChar - 'a';
    int row = std::stoi(rowStr) - 1;
    return row * BOARD_SIZE + col;
}

void printHexBoard(const Position& pos, const std::string& evalInfo) {
    std::cout << "   ";
    for (int i = 0; i < BOARD_SIZE; ++i) std::cout << (char)('A' + i) << " ";

    // Print Evaluation Info next to the board header
    if (!evalInfo.empty()) {
        std::cout << "   [MoHex Eval: " << evalInfo << "]";
    }
    std::cout << "\n";

    pos.printPosition();
}

// --- ROBUST GTP ENGINE ---
class GtpEngine {
    int pipe_in[2];
    int pipe_out[2];
    pid_t pid;
    std::string name;
    std::string logFilePath;

public:
    GtpEngine(const std::string& cmdLine) {
        // Dynamic Name Extraction
        name = getBaseName(cmdLine);

        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) throw std::runtime_error("Pipe failed");

        std::stringstream ss;
        ss << "mohex_log_" << getpid() << "_" << std::this_thread::get_id() << ".txt";
        logFilePath = ss.str();

        pid = fork();
        if (pid < 0) throw std::runtime_error("Fork failed");

        if (pid == 0) {
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);

            int logFd = open(logFilePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (logFd >= 0) {
                dup2(logFd, STDERR_FILENO);
                close(logFd);
            } else {
                int devNull = open("/dev/null", O_WRONLY);
                dup2(devNull, STDERR_FILENO);
                close(devNull);
            }

            close(pipe_in[1]);
            close(pipe_out[0]);

            std::vector<std::string> argStrs = splitCommand(cmdLine);
            std::vector<char*> args;
            for (const auto& s : argStrs) args.push_back(const_cast<char*>(s.c_str()));
            args.push_back(nullptr);

            execvp(args[0], args.data());
            exit(127);
        } else {
            close(pipe_in[0]);
            close(pipe_out[1]);
        }
    }

    ~GtpEngine() {
        sendCommand("quit");
        close(pipe_in[1]);
        close(pipe_out[0]);
        waitpid(pid, nullptr, 0);
        unlink(logFilePath.c_str());
    }

    // Parse the temporary log file to find "Score X.XX"
    std::string findScoreInLogs() {
        std::ifstream logFile(logFilePath);
        if (!logFile.is_open()) return "N/A";

        std::string line;
        std::string lastScore = "";

        // Simple scan: Read entire file (it's short per game usually) and find last "Score"
        while (std::getline(logFile, line)) {
            size_t found = line.find("Score");
            if (found != std::string::npos) {
                // MoHex format: "Score 0.32"
                lastScore = line.substr(found);
            }
        }
        return lastScore.empty() ? "Unknown" : lastScore;
    }

    void printLogTail() {
        std::ifstream logFile(logFilePath);
        if (!logFile.is_open()) return;

        std::string line;
        std::vector<std::string> lines;
        while (std::getline(logFile, line)) {
            lines.push_back(line);
            if (lines.size() > 20) lines.erase(lines.begin());
        }

        std::cerr << "--- LAST ENGINE LOGS (" << name << ") ---" << std::endl;
        for (const auto& l : lines) std::cerr << "   " << l << std::endl;
        std::cerr << "----------------------------------------" << std::endl;
    }

    void sendCommand(const std::string& cmd) {
        std::string full = cmd + "\n";
        if (write(pipe_in[1], full.c_str(), full.size()) < 0) { }
    }

    std::string readResponse() {
        std::string response;
        char buffer[256];
        while (true) {
            ssize_t count = read(pipe_out[0], buffer, sizeof(buffer) - 1);
            if (count <= 0) break;
            buffer[count] = '\0';
            response += buffer;
            if (response.find("\n\n") != std::string::npos) break;
        }
        return response;
    }

    void init(int simLimit) {
        std::string r;
        sendCommand("name");
        r = readResponse();
        std::cout << r << std::endl;
        sendCommand("boardsize 11");
        r = readResponse();
        std::cout << r << std::endl;
        sendCommand("clear_board");
        r = readResponse();
        std::cout << r << std::endl;
        // if (simLimit > 0) {
        //     sendCommand("param_mohex max_games " + std::to_string(simLimit));
        //     readResponse(); // Consume potential error silently
        // }
        sendCommand("param_mohex swap_allowed 0");
        r = readResponse();
        std::cout << r << std::endl;
        // sendCommand("param_mohex max_threads 1");
        // readResponse();
        sendCommand("param_wolve swap_allowed 0");
        r = readResponse();
        std::cout << r << std::endl;

        // sendCommand("param_wolve max_threads 1");
        // readResponse();
        //
        // std::string cmd = "time_settings " + std::to_string((int)0.5) + " 0 0";
        // sendCommand(cmd);
        // readResponse();
        //
        // // Also try explicit param (for safety)
        // // MoHex:
        // sendCommand("param_mohex max_time " + std::to_string(0.5));
        // readResponse(); // Ignore error if not MoHex
        //
        // // Wolve:
        // sendCommand("param_wolve max_time " + std::to_string(0.5));
        // readResponse(); // Ignore error if not Wolve

    }

    std::string getName() const { return name; }
};

// --- GAME LOGIC ---

// Helper to peek at MoHex's evaluation without advancing the game state
std::string getMoHexEval(GtpEngine& mohex, int sideToMove) {
    std::string color = (sideToMove == 0) ? "black" : "white";

    // 1. Ask for a move (implies search/eval)
    mohex.sendCommand("genmove " + color);
    std::string resp = mohex.readResponse();

    if (resp.empty() || resp[0] != '=') return "Crash/Error";

    // 2. Extract score from the side-channel logs
    std::string score = mohex.findScoreInLogs();

    // 3. Undo the move so the actual game loop can proceed normally
    mohex.sendCommand("undo");
    mohex.readResponse();

    return score;
}

GameResult playSingleGame(GtpEngine& black, GtpEngine& white, int blackID, uint64_t seed, int simLimit, int gameId) {    Position pos(0);
    GameResult res;
    res.blackEngineID = blackID;
    std::vector<std::string> moveHistory;

    black.init(simLimit);
    white.init(simLimit);

    FastRand rng(seed);

    // --- OPENING PHASE ---
    for (int i = 0; i < OPENING_PLIES; ++i) {
        if (pos.getWinner() != -1) break;
        int move = pos.getRandomLegalMove(rng);
        pos.makeMove(move);

        std::string moveStr = moveToString(move);
        moveHistory.push_back(moveStr);
        std::string color = (i % 2 == 0) ? "black" : "white";

        black.sendCommand("play " + color + " " + moveStr);
        if (black.readResponse().empty()) { black.printLogTail(); res.crashed = true; return res; }

        white.sendCommand("play " + color + " " + moveStr);
        if (white.readResponse().empty()) { white.printLogTail(); res.crashed = true; return res; }
    }

    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "\n--- Start of Game (" << black.getName() << " vs " << white.getName() << ") ---" << std::endl;
        printHexBoard(pos, "Opening");
    }

    int moves = OPENING_PLIES;
    while (pos.getWinner() == -1 && moves < SAFETY_LIMIT) {
        GtpEngine& currentEngine = (pos.sideToMove == 0) ? black : white;
        GtpEngine& otherEngine   = (pos.sideToMove == 0) ? white : black;
        std::string colorStr     = (pos.sideToMove == 0) ? "black" : "white";

        // --- EVALUATION STEP ---
        // We query the SECOND engine (assumed to be MoHex or similar strong engine)
        // to check the score. In 'playSingleGame', we don't know which is MoHex easily
        // unless we track names. Let's just ask the 'white' engine if blackID==0 (A vs MoHex),
        // or 'black' engine if blackID==1 (MoHex vs A).
        // SIMPLIFICATION: Just query 'otherEngine' if it's not the current mover, or 'current' if it is.
        // Actually, let's always query the engine known as "mohex" if possible.
        // Fallback: Query 'white' engine (Engine B) always for consistency?
        // Let's query WHOEVER IS ENGINE B (The reference engine).

        // Identify Engine B reference
        GtpEngine& refEngine = (blackID == 0) ? white : black;
        std::string evalInfo; //= getMoHexEval(refEngine, pos.sideToMove);

        // --- MOVE GENERATION ---
        currentEngine.sendCommand("genmove " + colorStr);
        std::string resp = currentEngine.readResponse();

        if (resp.empty() || resp[0] != '=') {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "[!] " << currentEngine.getName() << " CRASHED." << std::endl;
            currentEngine.printLogTail();
            res.winner = (pos.sideToMove == 0) ? 1 : 0;
            res.finalString = currentEngine.getName() + " Forfeited (Crash)";
            res.crashed = false;
            saveGameToSGF("game_" + std::to_string(gameId) + ".sgf",
                          black.getName(), white.getName(),
                          (pos.sideToMove == 0) ? 1 : 0, moveHistory);
            return res;
        }

        std::stringstream ss(resp.substr(1));
        std::string moveStr;
        ss >> moveStr;
        moveHistory.push_back(moveStr);
        if (moveStr == "resign") {
            res.winner = (pos.sideToMove == 0) ? 1 : 0;
            res.finalString = currentEngine.getName() + " resigned";
            saveGameToSGF("game_" + std::to_string(gameId) + ".sgf",
                          black.getName(), white.getName(),
                          (pos.sideToMove == 0) ? 1 : 0, moveHistory);
            return res;
        }

        int move = stringToMove(moveStr);
        if (!pos.isMoveLegal(move)) {
            res.crashed = true;
            return res;
        }

        pos.makeMove(move);
        moves++;

        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "\nMove " << moves << " | " << currentEngine.getName() << " (" << colorStr << ") played " << moveStr << ":" << std::endl;
            printHexBoard(pos, evalInfo);
        }

        otherEngine.sendCommand("play " + colorStr + " " + moveStr);
        if (otherEngine.readResponse().empty()) {
            otherEngine.printLogTail();
            res.crashed = true;
            return res;
        }
    }

    if (pos.getWinner() != -1) {
        res.winner = pos.getWinner();
        res.finalString = "Checkmate";
    } else {
        res.winner = -1;
        res.finalString = "Move Limit";
    }
    saveGameToSGF("game_" + std::to_string(gameId) + ".sgf",
                          black.getName(), white.getName(),
                          (pos.sideToMove == 0) ? 1 : 0, moveHistory);
    return res;
}

void worker(std::string cmdA, std::string cmdB, int pairsToPlay, int simLimit, std::atomic<int>& pairsFinished, Stats& stats) {
    for (int i = 0; i < pairsToPlay; ++i) {
        uint64_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                        + std::chrono::high_resolution_clock::now().time_since_epoch().count();

        // Game 1: A vs B
        int currentPair = pairsFinished.load();
        try {
            GtpEngine engineA(cmdA);
            GtpEngine engineB(cmdB);

            GameResult r1 = playSingleGame(engineA, engineB, 0, seed, simLimit, currentPair * 2);

            std::lock_guard<std::mutex> lock(print_mutex);
            if (r1.crashed) {
                stats.crashes++;
                std::cout << ">>> GAME CRASHED <<<" << std::endl;
            } else {
                stats.gamesPlayed++;
                std::cout << ">>> RESULT: " << r1.finalString << (r1.winner==0?" (Black Wins)":" (White Wins)") << std::endl;
                if (r1.winner == 0) stats.winsA++;
                else if (r1.winner == 1) stats.winsB++;
                else stats.draws++;
            }
        } catch (...) { stats.crashes++; }

        // Game 2: B vs A
        try {
            GtpEngine engineA(cmdA);
            GtpEngine engineB(cmdB);

            GameResult r2 = playSingleGame(engineB, engineA, 1, seed, simLimit, currentPair * 2 +1);

            std::lock_guard<std::mutex> lock(print_mutex);
            if (r2.crashed) {
                stats.crashes++;
                std::cout << ">>> GAME CRASHED <<<" << std::endl;
            } else {
                stats.gamesPlayed++;
                std::cout << ">>> RESULT: " << r2.finalString << (r2.winner==0?" (Black Wins)":" (White Wins)") << std::endl;
                if (r2.winner == 0) stats.winsB++;
                else if (r2.winner == 1) stats.winsA++;
                else stats.draws++;
            }
        } catch (...) { stats.crashes++; }

        pairsFinished++;
    }
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);

    if (argc < 3) {
        std::cerr << "Usage: ./Arbiter <AgentA_Cmd> <AgentB_Cmd> [NumPairs] [Threads] [SimLimit]" << std::endl;
        return 1;
    }

    std::string cmdA = argv[1];
    std::string cmdB = argv[2];
    int numPairs = (argc > 3) ? std::stoi(argv[3]) : 1;
    int numThreads = (argc > 4) ? std::stoi(argv[4]) : 1;
    int simLimit = (argc > 5) ? std::stoi(argv[5]) : 1000;

    std::cout << "=== VISUAL HEX TOURNAMENT ===" << std::endl;
    std::cout << "Agent A: " << getBaseName(cmdA) << std::endl;
    std::cout << "Agent B: " << getBaseName(cmdB) << std::endl;
    std::cout << "Pairs: " << numPairs << std::endl;
    std::cout << "-----------------------------" << std::endl;

    Stats globalStats;
    std::vector<std::thread> threads;
    std::atomic<int> pairsFinished{0};

    int pairsPerThread = numPairs / numThreads;
    int extra = numPairs % numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int task = pairsPerThread + (i < extra ? 1 : 0);
        if (task > 0) {
            threads.emplace_back(worker, cmdA, cmdB, task, simLimit, std::ref(pairsFinished), std::ref(globalStats));
        }
    }

    for (auto& t : threads) t.join();

    std::cout << "----------------------" << std::endl;
    std::cout << "Total Games: " << globalStats.gamesPlayed << std::endl;
    std::cout << "Agent A Wins: " << globalStats.winsA << std::endl;
    std::cout << "Agent B Wins: " << globalStats.winsB << std::endl;

    return 0;
}