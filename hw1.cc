/*
 * ALGO:
 * Q <- init_state
 * visited.Add(init_state.position)
 * while Q is not empty:
 *     curr_state <- Q.pop()
 *     for [key, dx, dy] in 4 directions:
 *         new_state <- computeState(curr_state, dx, dy)
 *         if isSolved(new_state):
 *             return moves.Get(curr_state) + key
 *         if !visited[new_state.bp_pos] && testState(new_state):
 *             Q.push(new_state)
 *             visited.Add(new_state.position)
 *             moves.Add(curr_state, new_state, key)
 *
 *
 * DS:
 * State:
 *     game_map: vector<strings>
 *     boxes & player position (position): vector<pair<int, int>>
 *     hash_key: A string converted from `position`.
 *
 * visited: An unordered_set of strings.
 *
 * moves: A trie.
 */

#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class State {
public:
    State() {}
    State(std::vector<std::string> game_map)
    {
    }

    std::vector<std::string> game_map;
    std::vector<std::pair<int, int>> position;
    std::string hash_key;
};

class Direction {
public:
    Direction(int key_, int dx_, int dy_)
    {
        key = key_;
        dx = dx_;
        dy = dy_;
    }

    char key;
    int dx, dy;
};

int main(int argc, char* argv[])
{
    std::string line;
    std::ifstream testcase(argv[1]);
    std::vector<std::string> game_map;

    if (testcase.is_open()) {
        while (std::getline(testcase, line)) {
            game_map.emplace_back(line);
        }
        testcase.close();
    } else
        std::cerr << "Unable to open file.\n";

    std::vector<Direction> dir = {
        { 'W', -1, 0 },
        { 'A', 0, -1 },
        { 'S', 1, 0 },
        { 'D', 0, 1 }
    };
    std::queue<State> Q;
    std::unordered_set<std::string> visited;

    return 0;
}
