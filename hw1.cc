/*
 * ALGO:
 * Q <- init_state
 * visited.insert(init_state.position)
 * while Q is not empty:
 *     curr_state <- Q.pop()
 *     for [key, dx, dy] in 4 directions:
 *         new_state <- computeState(curr_state, dx, dy)
 *         if new_state.valid:
 *             if isSolved(new_state):
 *                 return moves.Get(curr_state) + key
 *             if !visited.contain(new_state) && testState(new_state):
 *                 Q.push(new_state)
 *                 visited.insert(new_state.position)
 *                 moves.Add(curr_state, new_state, key)
 *
 * computeState: game_map, player_pos, dx, dy
 *     1. Get new player position
 *     2. Calculate virtual box position
 *     3. Update game_map:
 *         1. if game_map[new player pos] is ' ': Update to 'o'
 *            else if game_map[new player pos] is '.': Update to 'O'
 *            else if game_map[new player pos] is 'x' or 'X': 
 *                if game_map[virtual box pos] is ' ': Update to 'x'
 *                else if game_map[virtual box pos] is '.': Update to 'X'
 *                else return invalid state
 *                if game_map[new player pos] is 'x': Update to 'o'
 *                else: Update game_map[new player pos] to 'o'
 *            else return invalid state
 *         2. Update game_map[player pos]:
 *             if game_map[player pos] is 'o': Update to ' '
 *             else: Update to '.'
 *     4. Return new state of (game_map,  new player pos)
 *
 * isSolved: game_map
 *     Check if there is '.' in game_map
 *
 * DS:
 * State:
 *     game_map: vector<strings>
 *     player position: pair<int, int>
 *     boxes position: vector<pair<int, int>>
 *     target position: vector<pair<int, int>>
 *
 * visited: 
 *     history: An unordered_set of strings.
 *     Check if visited: vector v
 *         std::ostringstream os;
 *         for (const auto& p : v) {
 *             os << '-' << p.first << '-' << p.second;
 *         }
 *         hash_key = os.str();
 *         return (visited.find(hash_key) != visited.end());
 *
 * moves: A trie.
 */

#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

struct State {
    State()
    {
        isValid = false;
    }

    State(std::vector<std::string> game_map_, std::vector<std::pair<int, int>> pos_)
    {
        game_map = game_map_;
        position = pos_;
        isValid = true;

        std::ostringstream os;
        for (const auto& p : pos_) {
            os << p.first << '-' << p.second << '-';
        }
        hash_key = os.str();
    }

    std::vector<std::string> game_map;
    std::vector<std::pair<int, int>> position;
    std::string hash_key;
    bool isValid;
};

struct Direction {
    Direction(int key_, int dx_, int dy_)
    {
        key = key_;
        dx = dx_;
        dy = dy_;
    }

    char key;
    int dx, dy;
};

struct Record {
    bool contain(const State& state)
    {
        return (history.find(state.hash_key) != history.end());
    }

    void insert(const State& state)
    {
        history.insert(state.hash_key);
    }

    std::unordered_set<std::string> history;
};

State computeState(const State&, int, int);
bool isSolved(const State&);
bool testState(const State&);

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

    std::queue<State> Q;
    Record visited;
    Direction direction[4] = {
        { 'W', -1, 0 }, { 'A', 0, -1 }, { 'S', 1, 0 }, { 'D', 0, 1 }
    };

    while (!Q.empty()) {
        State curr_state = Q.front();
        Q.pop();

        for (int i = 0; i < 4; i++) {
            Direction dir = direction[i];
            State new_state = computeState(curr_state, dir.dx, dir.dy);

            if (new_state.isValid) {
            }
        }
    }

    return 0;
}
