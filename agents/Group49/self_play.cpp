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
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <cstring>
#include <algorithm>
#include <fcntl.h>
#include <csignal>
#include <memory>
#include <random>
#include <chrono>

#include "src/Position.h"
#include "src/Util.h"
#include "src/MCTS.h"
#include "src/InferenceServer.h"

using namespace engine;

// --- CONFIGURATION ---
// const std::string MOHEX_PATH =
    // "/home/d4k3rz/benzene-vanilla-cmake/build/src/mohex/mohex";
    // "mohex";
// const std::string MOHEX_CONFIG =
    // "/home/skynet/git/benzene-vanilla-cmake/mohex_selfplay.htp";
    // "mohex_selfplay.htp";

const std::string KATAHEX_PATH = "/home/skynet/git/katahex/build/katahex"; // Или полный путь /home/user/...
const std::string KATAHEX_CONFIG = "/home/skynet/git/katahex/config.cfg";
const std::string KATAHEX_MODEL = "/home/skynet/git/katahex/hex27x3.bin.gz";

const int TEMP_THRESHOLD = 20;

std::mutex io_mutex;

enum class Mode {
    MOHEX,
    AGENT
};

struct Sample {
    int playerToMove;
    std::array<uint8_t, BOARD_AREA> red;
    std::array<uint8_t, BOARD_AREA> blue;
    std::array<uint8_t, BOARD_AREA> turn;
    std::array<uint8_t, BOARD_AREA> last_move;
    std::array<uint8_t, BOARD_AREA> conn_start;
    std::array<uint8_t, BOARD_AREA> conn_end;
    std::array<double, BOARD_AREA> policy;
    float rootValue = 0.0f;
};


// --- HELPER FUNCTIONS ---

std::array<uint8_t, BOARD_AREA> extractPlane(const std::vector<float>& tensor, int channelIdx) {
    std::array<uint8_t, BOARD_AREA> plane{};
    int offset = channelIdx * BOARD_AREA;
    for (int i = 0; i < BOARD_AREA; ++i) {
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

int stringToMove(std::string s) {
    if (s.length() < 2) return -1;
    char colChar = std::tolower(s[0]);
    std::string rowStr = s.substr(1);
    int col = colChar - 'a';
    int row = std::stoi(rowStr) - 1;
    return row * 11 + col;
}

std::string moveToString(int move) {
    if (move < 0) return "resign";
    int row = move / BOARD_SIZE;
    int col = move % BOARD_SIZE;
    std::stringstream ss;
    ss << (char)('a' + col) << (row + 1);
    return ss.str();
}

int pickMoveFromPolicy(const std::array<double, BOARD_AREA>& policy, double temperature, FastRand& rng) {
    if (temperature < 0.01) {
        int bestMove = -1;
        double maxP = -1.0;
        for (int i=0; i<BOARD_AREA; ++i) {
            if (policy[i] > maxP) {
                maxP = policy[i];
                bestMove = i;
            }
        }
        return bestMove;
    }
    double r = (rng.range(10000) / 10000.0);
    double cumulative = 0.0;
    for (int i=0; i<BOARD_AREA; ++i) {
        if (policy[i] > 1e-9) {
            cumulative += policy[i];
            if (r <= cumulative) return i;
        }
    }
    int bestMove = -1;
    double maxP = -1.0;
    for (int i=0; i<BOARD_AREA; ++i) {
        if (policy[i] > maxP) { maxP = policy[i]; bestMove = i; }
    }
    return bestMove;
}

// [NEW] SGF Saver
void saveGameToSGF(const std::string& filename,
                   const std::string& blackName,
                   const std::string& whiteName,
                   int winner,
                   const std::vector<std::string>& moves)
{
    std::ofstream file(filename);
    std::cout << "Saving game to " << filename << std::endl;
    if (!file.is_open()) return;
    file << "(;FF[4]GM[11]SZ[11]\n";
    file << "PB[" << blackName << "]PW[" << whiteName << "]\n";

    if (winner == 0) file << "RE[B+Resign]\n";
    else if (winner == 1) file << "RE[W+Resign]\n";
    else file << "RE[Draw]\n";

    file << "DT[" << __DATE__ << "]\n";

    for (size_t i = 0; i < moves.size(); ++i) {
        char player = (i % 2 == 0) ? 'B' : 'W';
        file << ";" << player << "[" << moves[i] << "]\n";
    }

    file << ")\n";
    file.close();
    std::cout << "Saved" << filename << std::endl;
}

class GtpEngine {
    int pipe_in[2];
    int pipe_out[2];
    pid_t pid;

public:
    GtpEngine(const std::string& cmd, const std::string& configPath, const std::string& modelPath) {
        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0)
            throw std::runtime_error("Failed to create pipes");

        pid = fork();
        if (pid < 0)
            throw std::runtime_error("Failed to fork");

        if (pid == 0) {
            // Child
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);

            int devNull = open("/dev/null", O_WRONLY);
            dup2(devNull, STDERR_FILENO);
            close(devNull);

            close(pipe_in[1]);
            close(pipe_out[0]);

            execl(
                cmd.c_str(),        // Executable path
                "katahex",          // argv[0]
                "gtp",              // argv[1]
                "-config",          // argv[2]
                configPath.c_str(), // argv[3]
                "-model",           // argv[4]
                modelPath.c_str(),  // argv[5]
                nullptr             // End
            );

            perror("Execl failed");
            exit(127);
        } else {
            // Parent
            close(pipe_in[0]);
            close(pipe_out[1]);
        }
    }

    ~GtpEngine() {
        sendCommand("quit");
        close(pipe_in[1]);
        close(pipe_out[0]);
        waitpid(pid, nullptr, 0);
    }

    void sendCommand(const std::string& cmd) {
        std::string full_cmd = cmd + "\n";
        if (write(pipe_in[1], full_cmd.c_str(), full_cmd.size()) < 0) {}
    }

    std::string readResponse() {
        std::string response;
        char buffer[128];
        while (true) {
            ssize_t count = read(pipe_out[0], buffer, sizeof(buffer) - 1);
            if (count <= 0) break;
            buffer[count] = '\0';
            response += buffer;
            if (response.find("\n\n") != std::string::npos) break;
        }
        return response;
    }

    // [UPDATED] Clean readLine that strips newline chars
    std::string readLine() {
        std::string line;
        char c;
        while (read(pipe_out[0], &c, 1) > 0) {
            if (c == '\n' || c == '\r') {
                if (!line.empty()) break; // Return line if we have data
                continue; // Skip leading/duplicate newlines
            }
            line += c;
        }
        return line;
    }

    int getMove(int sideToMove) {
        std::string color = (sideToMove == 0) ? "black" : "white";
        sendCommand("genmove " + color);
        std::string resp = readResponse();
        std::cout << resp << std::endl;

        if (resp.empty() || resp[0] != '=') {
            std::cerr << "Error reading response from engine: " << resp << std::endl;
            return -1; // protocol error
        }
        std::stringstream ss(resp.substr(1));
        std::string moveStr;
        ss >> moveStr;

        if (moveStr == "resign") return -2;
        if (moveStr == "pass")   return -4; // PASS detected
        if (moveStr == "swap")   return -3;

        return stringToMove(moveStr);
    }

    void init(int seed) {
        // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        // sendCommand("boardsize 11");
        // std::cout << "doska gotova" << std::endl;
        // sendCommand("genmove b");
        // std::string resp = readResponse();
        // std::cout << resp << std::endl;
        // sendCommand("genmove w");
        // resp = readResponse();
        // std::cout << resp << std::endl;
        // sendCommand("genmove w");
        // resp = readResponse();
        // std::cout << resp << std::endl;
        // sendCommand("genmove b");
        // resp = readResponse();
        // std::cout << resp << std::endl;
        // sendCommand("showboard");
        // resp = readResponse();
        // std::cout << resp << std::endl;

        // Читаем всё подряд, пока не найдем ответ, начинающийся с '='.
        // Это проигнорирует "KataGo v1.12...", "WARNING..." и прочий мусор при старте.
        sendCommand("boardsize 11");
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        readResponse();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        sendCommand("clear_board");
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        readResponse();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        // Increase visits slightly for better generation quality
        sendCommand("kata-set-param maxVisits 100");
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        readResponse();
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
};

struct GameSamples {
    std::vector<Sample> samples;
    std::vector<std::string> moveHistory;
    int winner = -1;
};

std::string extractBestNonPassMove(const std::string& analysisLine) {
    std::string marker = "info move ";
    size_t pos = 0;

    while ((pos = analysisLine.find(marker, pos)) != std::string::npos) {
        pos += marker.length();
        size_t end = analysisLine.find(' ', pos);
        if (end == std::string::npos) end = analysisLine.length();

        std::string move = analysisLine.substr(pos, end - pos);
        if (move != "pass") {
            return move;
        }
    }
    return "";
}

// --- GAME LOOP: MOHEX ---
// --- GAME LOOP: MOHEX ---
GameSamples playMohexGame(GtpEngine& engine) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                      + std::chrono::high_resolution_clock::now().time_since_epoch().count();
    engine.init(static_cast<int>(seed));

    FastRand rng(seed);

    // Random Opening Phase
    int openingMoves = 1;
    for (int i = 0; i < openingMoves; ++i) {
        if (pos.getWinner() != -1) break;
        int randomMove = pos.getRandomLegalMove(rng);
        if (randomMove == -1) break;

        std::string color = (pos.sideToMove == 0) ? "black" : "white";
        record.moveHistory.push_back(moveToString(randomMove));

        pos.makeMove(randomMove);
        engine.sendCommand("play " + color + " " + moveToString(randomMove));
        if (engine.readResponse().empty()) throw std::runtime_error("Engine sync fail");
    }

    while (pos.getWinner() == -1) {
        // This implicitly plays the move on the engine if not passed/resigned
        int bestMove = engine.getMove(pos.sideToMove);

        // --- RESIGN ---
        if (bestMove == -2) {
            record.winner = 1 - pos.sideToMove;
            break;
        }

        // --- SWAP ---
        if (bestMove == -3) {
            record.moveHistory.push_back("swap");
            pos.moveCount++;
            continue;
        }

        // --- PASS (Forced Win Detected) ---
        if (bestMove == -4) {
            // 1. Undo the 'pass' so we can analyze the position
            engine.sendCommand("undo");
            engine.readResponse(); // Consumes "= \n\n"

            std::string color = (pos.sideToMove == 0) ? "black" : "white";

            // 2. Force analysis to find the physical move
            // We use '50' interval. The engine will acknowledge with "=" then stream "info..."
            engine.sendCommand("lz-analyze " + color + " 50");

            // 3. Robust Read Loop
            // We read lines until we find one starting with "info move"
            std::string messyLine;
            int maxAttempts = 100; // Safety break

            for(int k=0; k<maxAttempts; ++k) {
                std::string line = engine.readLine();

                // Skip empty lines or just "=" responses
                if (line.empty() || line == "=") continue;

                if (line.find("info move") != std::string::npos) {
                    messyLine = line;
                    break;
                }
            }

            // 4. Stop analysis immediately
            engine.sendCommand("stop");
            engine.readResponse(); // Consume the "= " response from stop

            // 5. Extract the best move
            std::string forcedMoveStr = extractBestNonPassMove(messyLine);

            if (!forcedMoveStr.empty()) {
                // Log the override
                // std::cout << "[SelfPlay] Overriding pass with " << forcedMoveStr << std::endl;

                // 6. Play the forced move on the engine
                engine.sendCommand("play " + color + " " + forcedMoveStr);

                // Read response for play command
                if (engine.readResponse().empty()) {
                    std::cerr << "Engine sync fail after override" << std::endl;
                    record.winner = 2; // Error
                    break;
                }

                // 7. Update bestMove integer so the rest of the loop proceeds
                bestMove = stringToMove(forcedMoveStr);
            } else {
                // If we genuinely can't find a move (shouldn't happen), treat as resign/loss
                std::cerr << "[Error] Could not extract move from: " << messyLine << std::endl;
                record.winner = 1 - pos.sideToMove;
                break;
            }
        }
        else if (bestMove < 0 || bestMove >= BOARD_AREA) {
            std::cerr << "Invalid move received: " << bestMove << std::endl;
            record.winner = 2; // Error
            break;
        }

        // --- RECORD SAMPLE (Standard Logic) ---
        Sample sample;
        sample.playerToMove = pos.sideToMove;
        std::vector<float> tensor = pos.toTensor();
        sample.red        = extractPlane(tensor, 0);
        sample.blue       = extractPlane(tensor, 1);
        sample.turn       = extractPlane(tensor, 2);
        sample.last_move  = extractPlane(tensor, 3);
        sample.conn_start = extractPlane(tensor, 4);
        sample.conn_end   = extractPlane(tensor, 5);
        sample.policy.fill(0.0);
        sample.policy[bestMove] = 1.0;
        sample.rootValue = 0.0f;

        pos.makeMove(bestMove);

        record.samples.push_back(sample);
        record.moveHistory.push_back(moveToString(bestMove));
    }

    if (record.winner == -1)
        record.winner = pos.getWinner();

    if (record.winner == -1)
        record.winner = 2; // error

    for (auto& sample : record.samples) {
        sample.rootValue =
            (record.winner == 2) ? 0.0f :
            (record.winner == sample.playerToMove ? 1.0f : -1.0f);
    }

    return record;
}

// --- GAME LOOP: AGENT (UPDATED) ---
GameSamples playAgentGame(MCTS& agent, InferenceServer& server, int simulations) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                      + std::chrono::high_resolution_clock::now().time_since_epoch().count();
    FastRand rng(seed);

    int openingMoves = 0;
    for(int i=0; i<openingMoves; ++i) {
         if (pos.getWinner() != -1) break;
         int randomMove = pos.getRandomLegalMove(rng);
         if(randomMove == -1) break;

         // FIX: Record history for random openings
         record.moveHistory.push_back(moveToString(randomMove));
         pos.makeMove(randomMove);
    }

    int movesPlayed = 0;
    while (pos.getWinner() == -1) {
        SearchResult result = agent.searchWithPolicy(pos, server, simulations);

        if (pos.moveCount == 1) {
            // If we (Blue) have < 50% win rate, Red's opening was too strong. Swap.
            // Note: result.rootValue is from the perspective of the side to move.
            if (result.rootValue < 0.0f) {
                // Execute Swap
                record.moveHistory.push_back("swap");
                pos.moveCount++;
                movesPlayed++;
                continue; // Skip making a physical move
            }
        }

        double temp = (movesPlayed < TEMP_THRESHOLD) ? 1.0 : 0.0;
        int chosenMove = pickMoveFromPolicy(result.policy, temp, rng);

        if (chosenMove < 0) break;

        Sample sample;
        sample.playerToMove = pos.sideToMove;
        std::vector<float> tensor = pos.toTensor();
        sample.red        = extractPlane(tensor, 0);
        sample.blue       = extractPlane(tensor, 1);
        sample.turn       = extractPlane(tensor, 2);
        sample.last_move  = extractPlane(tensor, 3);
        sample.conn_start = extractPlane(tensor, 4);
        sample.conn_end   = extractPlane(tensor, 5);
        sample.policy = result.policy;
        sample.rootValue = result.rootValue;

        record.samples.push_back(sample);

        // FIX: Record history for main game loop
        record.moveHistory.push_back(moveToString(chosenMove));

        pos.makeMove(chosenMove);
        movesPlayed++;
    }

    record.winner = pos.getWinner();
    if (record.winner == -1) record.winner = 2;
    return record;
}

// --- WORKER THREAD ---
// --- WORKER THREAD ---
void worker(Mode mode, int totalGames, int simulations, bool saveSGF, std::atomic<int>& gamesPlayed, std::ofstream& out, const std::string& modelPath) {
    std::unique_ptr<Inference> net;
    std::unique_ptr<InferenceServer> server;
    std::unique_ptr<MCTS> mcts_agent;

    // [OPTIMIZATION] Initialize KataHex engine once per thread if in MoHex mode
    // This prevents the expensive process creation/model load for every single game.
    std::unique_ptr<GtpEngine> sharedEngine;
    if (mode == Mode::MOHEX) {
        // Ensure you use the KATAHEX constants here as discussed for speed
        sharedEngine = std::make_unique<GtpEngine>(KATAHEX_PATH, KATAHEX_CONFIG, KATAHEX_MODEL);
    }

    if (mode == Mode::AGENT) {
        net = std::make_unique<Inference>(modelPath);
        server = std::make_unique<InferenceServer>(*net);
        mcts_agent = std::make_unique<MCTS>();
        mcts_agent->cpuct = 1.5;
    }

    while (true) {
        int gameIdx = gamesPlayed.fetch_add(1);
        if (gameIdx >= totalGames) return;

        try {
            GameSamples record;

            if (mode == Mode::MOHEX) {
                // Reuse the persistent engine.
                // playMohexGame calls init() which clears the board, so this is safe.
                record = playMohexGame(*sharedEngine);
            } else {
                record = playAgentGame(*mcts_agent, *server, simulations);
            }

            // --- CRITICAL SECTION: FILE I/O ---
            std::lock_guard<std::mutex> lock(io_mutex);

            // 1. Save SGF (Optional)
            if (saveSGF) {
                std::string fname = "game_" + std::to_string(gameIdx) + ".sgf";
                std::string bName = (mode == Mode::MOHEX) ? "MoHex" : "Agent";
                std::string wName = (mode == Mode::MOHEX) ? "MoHex" : "Agent";
                saveGameToSGF(fname, bName, wName, record.winner, record.moveHistory);
            }

            // 2. Save JSONL Data
            for (size_t moveIdx = 0; moveIdx < record.samples.size(); ++moveIdx) {
                const Sample& sample = record.samples[moveIdx];
                int value = 0;
                // Determine Value: 1 (Win), -1 (Loss), 0 (Draw/Error)
                if (record.winner != 2) {
                    // std::cerr << "Record winner: " << record.winner<< "playertomove"<< sample.playerToMove << std::endl;
                    value = (record.winner == sample.playerToMove) ? 1 : -1;
                }

                out << '{'
                    << "\"game\":" << gameIdx
                    << ",\"move\":" << moveIdx
                    << ",\"player\":" << sample.playerToMove
                    << ",\"value\":" << value
                    << ",\"root_value\":" << sample.rootValue
                    << ",\"red\":" << planeToJson(sample.red)
                    << ",\"blue\":" << planeToJson(sample.blue)
                    << ",\"turn\":" << planeToJson(sample.turn)
                    << ",\"last_move\":" << planeToJson(sample.last_move)
                    << ",\"conn_start\":" << planeToJson(sample.conn_start)
                    << ",\"conn_end\":" << planeToJson(sample.conn_end)
                    << ",\"policy\":" << policyToJson(sample.policy)
                    << "}\n";
            }

            // 3. [NEW] Flush to disk every 100 games
            // 'static' ensures this count is shared across all calls/threads (protected by io_mutex)
            static int gamesSavedCount = 0;
            gamesSavedCount++;

            if (gamesSavedCount % 100 == 0) {
                out.flush(); // Forces the OS to write the buffer to the physical file
                std::cout << "[Auto-Save] Flushed " << gamesSavedCount << " games to disk." << std::endl;
            }

            // Log progress
            if ((gameIdx + 1) % 10 == 0) { // Reduced log frequency slightly to reduce spam
                std::cout << "Finished " << (mode == Mode::MOHEX ? "MoHex" : "Agent")
                          << " game " << gameIdx + 1 << "/" << totalGames
                          << " (" << record.samples.size() << " moves)" << std::endl;
            }

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cerr << "[WARNING] Game " << gameIdx << " Failed: " << e.what() << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);

    if (argc < 2) {
        std::cerr << "Usage: ./SelfPlay <mode: mohex|agent> [games] [sims/file] [output_file] [save_sgf 0|1]" << std::endl;
        return 1;
    }

    std::string modeStr = argv[1];
    Mode mode = (modeStr == "agent") ? Mode::AGENT : Mode::MOHEX;

    int games = (argc > 2) ? std::stoi(argv[2]) : 10;
    int simulations = 512;
    std::string outputPath = (mode == Mode::AGENT) ? "agent_data.jsonl" : "mohex_data.jsonl";
    bool saveSGF = false;

    if (argc > 3) {
        if (mode == Mode::AGENT) simulations = std::stoi(argv[3]);
        else outputPath = argv[3];
    }
    if (argc > 4) {
        if (mode == Mode::AGENT) outputPath = argv[4];
        else saveSGF = (std::stoi(argv[4]) != 0);
    }
    if (argc > 5 && mode == Mode::AGENT) {
        saveSGF = (std::stoi(argv[5]) != 0);
    }
    std::string modelPath = "models/hex_run_6.onnx"; // Default
    if (mode == Mode::AGENT && argc > 6) {
        modelPath = argv[6]; // Allow overriding model path
    }

    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    unsigned int nThreads = 4;

    // if (mode == Mode::AGENT) nThreads = 6;

    std::cout << "Starting Self-Play | Mode: " << (mode == Mode::MOHEX ? "MOHEX" : "AGENT") << std::endl;
    std::cout << "Games: " << games << " | Threads: " << nThreads << " | SGF Logging: " << (saveSGF ? "ON" : "OFF") << std::endl;
    if (mode == Mode::AGENT) std::cout << "MCTS Simulations: " << simulations << std::endl;

    std::vector<std::thread> threads;
    std::atomic<int> gamesPlayed{0};

	auto t_start = std::chrono::steady_clock::now();

    for (unsigned int i = 0; i < nThreads; ++i) {
        threads.emplace_back(worker, mode, games, simulations, saveSGF, std::ref(gamesPlayed), std::ref(out),modelPath);
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

	auto t_end = std::chrono::steady_clock::now();
	double seconds = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Done. Saved to " << outputPath << std::endl;
	std::cout << "Self-play time: " << seconds << " seconds" << std::endl;
    return 0;
}