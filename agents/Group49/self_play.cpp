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
#include <fcntl.h> // <--- ADDED for O_WRONLY
#include <csignal> // <--- ADD THIS
#include "src/Position.h"
#include "src/Util.h"

using namespace engine;

// --- CONFIGURATION ---
const std::string MOHEX_PATH = "/home/julius/benzene-vanilla-cmake/build/src/mohex/mohex";

std::mutex io_mutex;

struct Sample {
    int playerToMove;
    std::array<uint8_t, BOARD_AREA> red;
    std::array<uint8_t, BOARD_AREA> blue;
    std::array<uint8_t, BOARD_AREA> turn;
    std::array<uint8_t, BOARD_AREA> last_move;
    std::array<uint8_t, BOARD_AREA> conn_start;
    std::array<uint8_t, BOARD_AREA> conn_end;
    std::array<double, BOARD_AREA> policy;
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

// --- GTP ENGINE WRAPPER ---
class GtpEngine {
    int pipe_in[2];
    int pipe_out[2];
    pid_t pid;

public:
    GtpEngine(const std::string& cmd) {
        if (pipe(pipe_in) < 0 || pipe(pipe_out) < 0) {
            throw std::runtime_error("Failed to create pipes");
        }

        pid = fork();
        if (pid < 0) {
            throw std::runtime_error("Failed to fork");
        }

        if (pid == 0) {
            // --- CHILD PROCESS (MoHex) ---

            // 1. Redirect Stdin/Stdout to pipes
            dup2(pipe_in[0], STDIN_FILENO);
            dup2(pipe_out[1], STDOUT_FILENO);

            // 2. Redirect Stderr to /dev/null
            // CRITICAL FIX: MoHex prints logs to stderr. If we pipe this to stdout,
            // our parser will read log messages instead of moves.
            int devNull = open("/dev/null", O_WRONLY);
            dup2(devNull, STDERR_FILENO);
            close(devNull);

            // 3. Close unused pipe ends
            close(pipe_in[1]);
            close(pipe_out[0]);

            execl(cmd.c_str(), cmd.c_str(), nullptr);
            exit(127);
        } else {
            // --- PARENT PROCESS ---
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
        write(pipe_in[1], full_cmd.c_str(), full_cmd.size());
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

    int getMove(int sideToMove) {
        // CRITICAL FIX: MoHex speaks GTP.
        // GTP only understands "black" and "white".
        // Player 0 (Red) maps to Black (First Player).
        // Player 1 (Blue) maps to White (Second Player).
        std::string color = (sideToMove == 0) ? "black" : "white";

        sendCommand("genmove " + color);

        std::string resp = readResponse();

        // CHECK FOR EMPTY (CRASH) FIRST
        if (resp.empty()) {
            std::cerr << "Engine Crashed (Empty Response)" << std::endl;
            return -1;
        }

        // Response format: "= C5\n\n"
        if (resp[0] != '=') {
            std::cerr << "Engine Protocol Error: " << resp << std::endl;
            return -1;
        }

        std::stringstream ss(resp.substr(1)); // Skip "= "
        std::string moveStr;
        ss >> moveStr;

        if (moveStr == "resign") return -1;

        // MoHex might return "swap". Treat as -1 (end game) or handle logic.
        if (moveStr == "swap") return -1;

        return stringToMove(moveStr);
    }

    void init(int seed) {
        sendCommand("boardsize 11");
        readResponse();
        sendCommand("clear_board");
        readResponse();
        sendCommand("param_mohex max_games 512");
        readResponse();
        sendCommand("param_mohex swap_allowed 0");
        readResponse();
        sendCommand("param_mohex random_seed " + std::to_string(seed));
        readResponse();
        sendCommand("param_mohex random_opening_moves 1"); // Example param if supported
        readResponse();
    }
};

struct GameSamples {
    std::vector<Sample> samples;
    int winner;
};
std::string moveToString(int move) {
    if (move < 0) return "resign";

    int row = move / BOARD_SIZE;
    int col = move % BOARD_SIZE;

    std::stringstream ss;
    ss << (char)('a' + col) << (row + 1);
    return ss.str();
}

GameSamples playMohexGame(GtpEngine& engine) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                      + std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Pass seed to init (truncate to int if needed by engine)
    engine.init(static_cast<int>(seed));
    if (true) { // Toggle this on/off
        // 1. Setup RNG once (outside the loop)
        FastRand rng(seed);

        // 2. Define how many random moves to play (e.g., 6 plies = 3 moves each)
        int openingMoves = 4;

        for (int i = 0; i < openingMoves; ++i) {
            // Check if game ended early during random phase
            if (pos.getWinner() != -1) break;

            // A. Pick Random Move
            int randomMove = pos.getRandomLegalMove(rng);
            if (randomMove == -1) break; // No legal moves left

            // B. Capture Sample (Policy = 100% on this random move)
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
            sample.policy[randomMove] = 1.0;

            record.samples.push_back(sample);

            // C. Determine Color String BEFORE updating local state
            // Player 0 = Black/Red, Player 1 = White/Blue
            std::string color = (pos.sideToMove == 0) ? "black" : "white";

            // D. Update Local State
            pos.makeMove(randomMove);

            // E. Sync with MoHex
            // We explicitly tell MoHex which color is playing to ensure safety
            engine.sendCommand("play " + color + " " + moveToString(randomMove));

            // F. Check for errors
            std::string resp = engine.readResponse();
            if (resp.empty() || resp[0] != '=') {
                // Handle crash or illegal move error
                throw std::runtime_error("MoHex rejected random move: " + resp);
            }
        }
    }

    while (pos.getWinner() == -1) {
        int bestMove = engine.getMove(pos.sideToMove);

        if (bestMove < 0 || bestMove >= BOARD_AREA) {
            break;
        }

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

        record.samples.push_back(sample);

        pos.makeMove(bestMove);
        // Note: We don't need to tell MoHex to "play" the move,
        // because "genmove" automatically applies the move to MoHex's internal board.
    }

    record.winner = pos.getWinner();
    if (record.winner == -1) record.winner = 2;

    return record;
}

void worker(int totalGames, std::atomic<int>& gamesPlayed, std::ofstream& out) {
    while (true) {
        int gameIdx = gamesPlayed.fetch_add(1);
        if (gameIdx >= totalGames) return;

        try {
            // --- CRITICAL FIX: Spawn Engine INSIDE the loop ---
            // If the previous game crashed, this creates a FRESH process.
            GtpEngine engine(MOHEX_PATH);

            GameSamples record = playMohexGame(engine);

            // Write to file (Thread-Safe)
            std::lock_guard<std::mutex> lock(io_mutex);

            // ... (Writing logic remains the same) ...
            for (size_t moveIdx = 0; moveIdx < record.samples.size(); ++moveIdx) {
                const Sample& sample = record.samples[moveIdx];
                int value = 0;
                if (record.winner != 2) {
                    value = (record.winner == sample.playerToMove) ? 1 : -1;
                }

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

            if ((gameIdx + 1) % 1 == 0) {
                std::cout << "Finished MoHex game " << gameIdx + 1 << "/" << totalGames
                          << " (" << record.samples.size() << " moves) [Thread "
                          << std::this_thread::get_id() << "]" << std::endl;
            }

        } catch (const std::exception& e) {
            // If spawning fails or engine crashes mid-setup, we catch it here.
            // The loop continues to the next game!
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cerr << "[WARNING] Game " << gameIdx << " Failed: " << e.what() << std::endl;
        }
    }
}
int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    int games = 10;
    std::string outputPath = "mohex_data.jsonl";

    if (argc > 1) games = std::stoi(argv[1]);
    if (argc > 2) outputPath = argv[2];

    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    unsigned int nThreads = 13;//std::thread::hardware_concurrency();
    if (nThreads > 1) nThreads -= 1;

    std::cout << "Starting MoHex Data Generation" << std::endl;
    std::cout << "Engine: " << MOHEX_PATH << std::endl;
    std::cout << "Games: " << games << " | Threads: " << nThreads << std::endl;

    std::vector<std::thread> threads;
    std::atomic<int> gamesPlayed{0};

    for (unsigned int i = 0; i < nThreads; ++i) {
        threads.emplace_back(worker, games, std::ref(gamesPlayed), std::ref(out));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    std::cout << "Done. Saved to " << outputPath << std::endl;
    return 0;
}