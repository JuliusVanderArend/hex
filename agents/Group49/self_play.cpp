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
const std::string MOHEX_PATH =
    "/Users/serhiitupikin/Documents/Coding/University/benzene-vanilla-cmake/build/src/mohex/mohex";
const std::string MOHEX_CONFIG =
    "/Users/serhiitupikin/Documents/Coding/University/hex/agents/Group49/mohex_selfplay.htp";

const int TEMP_THRESHOLD = 20;

std::mutex io_mutex;

enum class Mode {
    MOHEX,
    AGENT,
    AGENT_VS_MOHEX
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

// --- GTP ENGINE WRAPPER ---
class GtpEngine {
    int pipe_in[2];
    int pipe_out[2];
    pid_t pid;

public:
    GtpEngine(const std::string& cmd, const std::string& configPath) {
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
                cmd.c_str(),
                cmd.c_str(),
                ("--config=" + configPath).c_str(),
                nullptr
            );

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
        // std::cerr << full_cmd << std::endl;
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
    	std::string color = (sideToMove == 0) ? "black" : "white";
    	sendCommand("genmove " + color);
    	std::string resp = readResponse();
        // std::cerr << resp << std::endl;
    	if (resp.empty() || resp[0] != '=')
     	   return -1; // protocol error

   		std::stringstream ss(resp.substr(1));
    	std::string moveStr;
    	ss >> moveStr;

    	if (moveStr == "resign")
        	return -2;   // Resign value

    	if (moveStr == "swap" || moveStr == "swap-pieces")
        	return -3;   // Swap value

    	return stringToMove(moveStr);
}


    void init(int seed) {
    sendCommand("boardsize 11");
    readResponse();
    sendCommand("clear_board");
    readResponse();
    sendCommand("param_mohex random_seed " + std::to_string(seed));
    readResponse();
}
};

struct GameSamples {
    std::vector<Sample> samples;
    std::vector<std::string> moveHistory;
    int winner = -1;
};

// --- GAME LOOP: MOHEX ---
GameSamples playMohexGame(GtpEngine& engine) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                      + std::chrono::high_resolution_clock::now().time_since_epoch().count();
    engine.init(static_cast<int>(seed));

    FastRand rng(seed);
    int openingMoves = rng.range(3);  // No random opening moves
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
    	int bestMove = engine.getMove(pos.sideToMove);

    	// --- RESIGN ---
    	if (bestMove == -2) {
    	    std::cerr << "resign" << std::endl;
        	record.winner = 1 - pos.sideToMove;
        	break;
    	}

    	// --- Error (invalid move) ---
    	if (bestMove < 0 || bestMove >= BOARD_AREA) {
    	    std::cerr << "crash" << std::endl;
    	    record.winner = 1 - pos.sideToMove;
        	break;
    	}

    	// --- Normal move ---
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

    //pos.printPosition();

    if (record.winner == -1)
    	record.winner = pos.getWinner();

	if (record.winner == -1)
    	record.winner = 2; // error

	for (auto& sample : record.samples) {
    	sample.rootValue =
        	(record.winner == 2) ? 0.0f :
        	(record.winner == sample.playerToMove ? 1.0f : -1.0f);
	}

	std::cout << "Winner = " << pos.getWinner() << std::endl;
    std::cout << "Winner.winner = " << record.winner << std::endl;
    return record;
}

// --- GAME LOOP: AGENT (UPDATED) ---
GameSamples playAgentGame(MCTS& agent, InferenceServer& server, int simulations) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                      + std::chrono::high_resolution_clock::now().time_since_epoch().count();
    FastRand rng(seed);

    int openingMoves = rng.range(3); // Randomly pick 0, 1, or 2
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

// --- GAME LOOP: AGENT VS MOHEX ---
GameSamples playAgentVsMohexGame(
    MCTS& agent,
    InferenceServer& server,
    GtpEngine& mohex,
    int simulations,
    bool agentPlaysBlack  // true = agent is black, false = agent is white
) {
    Position pos(0);
    GameSamples record;

    size_t seed = std::hash<std::thread::id>{}(std::this_thread::get_id())
                  + std::chrono::high_resolution_clock::now().time_since_epoch().count();
    FastRand rng(seed);
    mohex.init(static_cast<int>(seed));

    int movesPlayed = 0;

    while (pos.getWinner() == -1) {
        // Determine whose turn it is
        // sideToMove: 0 = black, 1 = white
        bool isAgentTurn = (pos.sideToMove == 0) == agentPlaysBlack;

        int bestMove;

        if (isAgentTurn) {
            // Agent's turn - use MCTS with neural network
            SearchResult result = agent.searchWithPolicy(pos, server, simulations);

            // Handle swap rule for agent when playing as blue (move 1)
            if (pos.moveCount == 1 && !agentPlaysBlack) {
                if (result.rootValue < 0.0f) {
                    record.moveHistory.push_back("swap");
                    mohex.sendCommand("play white swap-pieces");
                    mohex.readResponse();
                    pos.moveCount++;
                    movesPlayed++;
                    continue;
                }
            }

            double temp = (movesPlayed < TEMP_THRESHOLD) ? 1.0 : 0.0;
            bestMove = pickMoveFromPolicy(result.policy, temp, rng);

            if (bestMove < 0) break;

            // Record sample for training data
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

            // Tell MoHex about agent's move
            std::string color = (pos.sideToMove == 0) ? "black" : "white";
            mohex.sendCommand("play " + color + " " + moveToString(bestMove));
            mohex.readResponse();
        } else {
            // MoHex's turn
            bestMove = mohex.getMove(pos.sideToMove);

            if (bestMove == -2) { // Resign
                std::cerr << "MoHex resigned" << std::endl;
                record.winner = 1 - pos.sideToMove;
                break;
            }

            if (bestMove == -3) { // Swap
                record.moveHistory.push_back("swap");
                pos.moveCount++;
                movesPlayed++;
                continue;
            }

            if (bestMove < 0 || bestMove >= BOARD_AREA) {
                std::cerr << "MoHex error/crash" << std::endl;
                record.winner = 1 - pos.sideToMove;
                break;
            }
        }

        record.moveHistory.push_back(moveToString(bestMove));
        pos.makeMove(bestMove);
        movesPlayed++;
    }

    if (record.winner == -1) record.winner = pos.getWinner();
    if (record.winner == -1) record.winner = 2;

    // Update sample values based on outcome
    for (auto& s : record.samples) {
        s.rootValue = (record.winner == 2) ? 0.0f
                    : (record.winner == s.playerToMove ? 1.0f : -1.0f);
    }

    std::cout << "Agent vs MoHex | Agent=" << (agentPlaysBlack ? "Black" : "White")
              << " | Winner=" << record.winner << std::endl;

    return record;
}

// --- WORKER THREAD ---
void worker(Mode mode, int totalGames, int simulations, bool saveSGF, std::atomic<int>& gamesPlayed, std::ofstream& out, const std::string& modelPath) {
    std::unique_ptr<Inference> net;
    std::unique_ptr<InferenceServer> server;
    std::unique_ptr<MCTS> mcts_agent;

    if (mode == Mode::AGENT || mode == Mode::AGENT_VS_MOHEX) {
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
                GtpEngine engine(MOHEX_PATH, MOHEX_CONFIG);
                record = playMohexGame(engine);
            } else if (mode == Mode::AGENT_VS_MOHEX) {
                GtpEngine engine(MOHEX_PATH, MOHEX_CONFIG);
                bool agentPlaysBlack = (gameIdx % 2 == 0);  // Alternate colors each game
                record = playAgentVsMohexGame(*mcts_agent, *server, engine, simulations, agentPlaysBlack);
            } else {
                record = playAgentGame(*mcts_agent, *server, simulations);
            }

            std::lock_guard<std::mutex> lock(io_mutex);

            if (saveSGF) {
                std::string fname = "game_" + std::to_string(gameIdx) + ".sgf";
                std::string bName, wName;
                if (mode == Mode::MOHEX) {
                    bName = "MoHex"; wName = "MoHex";
                } else if (mode == Mode::AGENT_VS_MOHEX) {
                    bool agentPlaysBlack = (gameIdx % 2 == 0);
                    bName = agentPlaysBlack ? "Agent" : "MoHex";
                    wName = agentPlaysBlack ? "MoHex" : "Agent";
                } else {
                    bName = "Agent"; wName = "Agent";
                }
                saveGameToSGF(fname, bName, wName, record.winner, record.moveHistory);
            }

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

            if ((gameIdx + 1) % 1 == 0) {
                const char* modeName = (mode == Mode::MOHEX) ? "MoHex"
                                     : (mode == Mode::AGENT_VS_MOHEX) ? "AgentVsMoHex"
                                     : "Agent";
                std::cout << "Finished " << modeName
                          << " game " << gameIdx + 1 << "/" << totalGames
                          << " (" << record.samples.size() << " moves) [Thread "
                          << std::this_thread::get_id() << "]" << std::endl;
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
        std::cerr << "Usage: ./SelfPlay <mode: mohex|agent|agent_vs_mohex> [games] [sims/file] [output_file] [save_sgf 0|1] [model_path]" << std::endl;
        return 1;
    }

    std::string modeStr = argv[1];
    Mode mode;
    if (modeStr == "agent") mode = Mode::AGENT;
    else if (modeStr == "agent_vs_mohex") mode = Mode::AGENT_VS_MOHEX;
    else mode = Mode::MOHEX;

    int games = (argc > 2) ? std::stoi(argv[2]) : 10;
    int simulations = 512;
    bool usesAgent = (mode == Mode::AGENT || mode == Mode::AGENT_VS_MOHEX);
    std::string outputPath = usesAgent ? "agent_data.jsonl" : "mohex_data.jsonl";
    if (mode == Mode::AGENT_VS_MOHEX) outputPath = "agent_vs_mohex_data.jsonl";
    bool saveSGF = false;

    if (argc > 3) {
        if (usesAgent) simulations = std::stoi(argv[3]);
        else outputPath = argv[3];
    }
    if (argc > 4) {
        if (usesAgent) outputPath = argv[4];
        else saveSGF = (std::stoi(argv[4]) != 0);
    }
    if (argc > 5 && usesAgent) {
        saveSGF = (std::stoi(argv[5]) != 0);
    }
    std::string modelPath = "models/hex_run_6.onnx"; // Default
    if (usesAgent && argc > 6) {
        modelPath = argv[6]; // Allow overriding model path
    }

    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Unable to open output file: " << outputPath << std::endl;
        return 1;
    }

    unsigned int nThreads = 10;
    // if (mode == Mode::AGENT) nThreads = 6;

    const char* modeNameLog = (mode == Mode::MOHEX) ? "MOHEX"
                             : (mode == Mode::AGENT_VS_MOHEX) ? "AGENT_VS_MOHEX"
                             : "AGENT";
    std::cout << "Starting Self-Play | Mode: " << modeNameLog << std::endl;
    std::cout << "Games: " << games << " | Threads: " << nThreads << " | SGF Logging: " << (saveSGF ? "ON" : "OFF") << std::endl;
    if (usesAgent) {
        std::cout << "MCTS Simulations: " << simulations << std::endl;
        std::cout << "Model Path: " << modelPath << std::endl;
    }

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