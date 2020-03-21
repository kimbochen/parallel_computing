#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::vector;
using std::string;
using std::queue;

class Coord {
public:
    Coord() {}

    Coord(int x_, int y_)
    {
        x = x_;
        y = y_;
    }

    Coord operator + (const Coord& c)
    {
        return Coord(x + c.x, y + c.y);
    }

    Coord operator * (const int k)
    {
        return Coord(k * x, k * y);
    }

    bool operator == (const Coord& c)
    {
        return (x == c.x && y == c.y);
    }

#ifdef DEBUG
    void debug() const
    {
        std::cout << '(' << x << ',' << y << ')';
    }
#endif

    int x, y;
};

struct GameMap {
    vector<string> tiles;
    vector<Coord> targetPos;
    vector<Coord> boxPos;
    Coord playerPos;

#ifdef DEBUG
    void debug() const
    {
        using std::cout;

        cout << "Map:\n";
        for (const auto& s : tiles) {
            cout << s << '\n';
        }

        cout << "Targets: ";
        for (const auto& t : targetPos) {
            t.debug();
            cout << ' ';
        }

        cout << "boxPos: ";
        for (const auto& c : boxPos) {
            c.debug();
            cout << ' ';
        }
        cout << '\n';

        cout << "playerPos: ";
        playerPos.debug();

        cout << "\n\n";
    }
#endif
};

struct Move {
    Move(char key_, int x, int y)
    {
        key = key_;
        dir = Coord(x, y);
    }

    char key;
    Coord dir;
};

class Node {
public:
    Node(Node* prevNode_, char key_)
    {
        prevNode = prevNode_;
        key = key_;
    }

    Node *prevNode;
    char key;
};

class State {
public:
    State() {}

    State(Coord& playerPos_, vector<Coord>& boxPos_, Node* prevNode = nullptr, char key = '\0')
    {
        std::ostringstream os;

        os << playerPos.x << playerPos.y;
        for (const auto& c : boxPos) {
            os << c.x << c.y;
        }

        hashKey = os.str();
        playerPos = playerPos_;
        boxPos = boxPos_;
        keyNode = new Node(prevNode, key);
    }

    State operator + (const Move& m)
    {
        Coord newPlayerPos = playerPos + m.dir;
        vector<Coord> newBoxPos = boxPos;

        for (auto& bp : newBoxPos) {
            if (bp == newPlayerPos) {
                bp = bp + m.dir;
                break;
            }
        }

        return State(newPlayerPos, newBoxPos, keyNode, m.key);
    }

#ifdef DEBUG
    void debug()
    {
        using std::cout;

        cout << "boxPos: ";
        for (const auto& c : boxPos) {
            c.debug();
            cout << ' ';
        }

        cout << "  playerPos: ";
        playerPos.debug();

        cout << " keyNode: " << keyNode->key << '\n';
    }
#endif

    Coord playerPos;
    vector<Coord> boxPos;
    string hashKey;
    Node* keyNode;
};

class Record {
public:
    bool contain(const State& state)
    {
        auto it = history.find(state.hashKey);
        return (it != history.end());
    }

    void insert(const State& state)
    {
        history.insert(state.hashKey);
    }

#ifdef DEBUG
    void debug()
    {
        std::cout << "Record: ";
        for (const auto& s : history) {
            std::cout << s << ' ';
        }
        std::cout << '\n';
    }
#endif

    std::unordered_set<std::string> history;
};

bool isValidMove(State& state, Move& m, GameMap& gameMap)
{
    Coord newPlayerPos = state.playerPos + m.dir;
    char playerTile = gameMap.tiles[newPlayerPos.x][newPlayerPos.y];

    if (playerTile != '#') {
        if (playerTile == 'x' || playerTile == 'X') {
            Coord newBoxPos = state.playerPos + m.dir * 2;
            char boxTile = gameMap.tiles[newBoxPos.x][newBoxPos.y];

            return (boxTile != '#');
        }
        else
            return true;
    }

    return false;
}

bool isSolution(State& state, GameMap& gameMap)
{
    int n = gameMap.targetPos.size();

    for (int i = 0; i < n; i++) {
        if (state.boxPos[i] == gameMap.targetPos[i])
            continue;
        else
            return false;
    }

    return true;
}

string getActionSequence(const State& state)
{
    Node *current = state.keyNode;
    string actionSeq;

    while (current != nullptr) {
        actionSeq.push_back(current->key);
        current = current->prevNode;
    }

    std::reverse(actionSeq.begin(), actionSeq.end());

    return actionSeq;
}

bool isDeadlock(const State& state, const GameMap& gameMap)
{
    return false;
}

string findActionSequence(GameMap& gameMap, vector<Node*> &nodes)
{
    State initState = State(gameMap.playerPos, gameMap.boxPos);
    Record visited;
    queue<State> Q;
    Move moves[4] = {
        { 'W', -1, 0 }, { 'A', 0, -1 }, { 'S', 1, 0 }, { 'D', 0, 1 }
    };
    std::unordered_map<std::string, std::string> actionSeq;
#ifdef STATE
    gameMap.debug();
    initState.debug();
    std::cout << "------------\n";
#endif

#ifdef GAME_MAP
    gameMap.debug();
#endif

#ifdef MOVE
    for (int i = 0; i < 4; i++) {
        Move m = moves[i];
        std::cout << m.key << ' ';
        m.dir.debug();
        std::cout << '\n';
    }
#endif

#ifdef STATE
    gameMap.debug();
    for (int i = 0; i < 4; i++) {
        Move m = moves[i];
        State newState = initState + m;
        std::cout << m.key << ' ';
        m.dir.debug();
        std::cout << ' ';
        newState.debug();
    }
#endif

#ifdef ISVALIDMOVE
    for (int i = 0; i < 4; i++) {
        Move m = moves[i];
        if (isValidMove(initState, m, gameMap)) {
            State newState = initState + m;
            std::cout << m.key << ' ';
            m.dir.debug();
            std::cout << ' ';
            newState.debug();
        }
    }
#endif

#ifdef ISSOLUTION
    for (int i = 0; i < 4; i++) {
        Move m = moves[i];
        State newState = initState + m;
        std::cout << m.key << ' ';
        m.dir.debug();
        std::cout << ' ';
        newState.debug();
        std::cout << isSolution(newState, gameMap) << '\n';
    }

    State testState(gameMap.playerPos, gameMap.targetPos);
    std::cout << "Test state: " << isSolution(testState, gameMap) << '\n';
#endif
    Q.emplace(initState);
    visited.insert(initState);
    actionSeq[initState.hashKey] = "";

    /*
    while (!Q.empty()) {
        State currState = Q.front();
        Q.pop();

        for (int i = 0; i < 4; i++) {
            Move m = moves[i];

            if (!isValidMove(currState, m, gameMap)) continue;

            State newState = currState + m;
            string newSeq = actionSeq[currState.hashKey] + m.key;

            if (!visited.contain(newState)) {
                nodes.emplace_back(newState.keyNode);

                if (isSolution(newState, gameMap))
                    // return getActionSequence(newState);
                    return newSeq;

                if (!isDeadlock(newState, gameMap)) {
                    Q.emplace(newState);
                    visited.insert(newState);
                    actionSeq[newState.hashKey] = newSeq;
                }
            }
            else delete newState.keyNode;
        }

    }
    */

#ifdef ALGO
    using std::cout;

    gameMap.debug();

    while (!Q.empty()) {
        State currState = Q.front();
        Q.pop();

        cout << "Current state:\n";
        currState.debug();

        for (int i = 0; i < 4; i++) {
            Move m = moves[i];

            if (!isValidMove(currState, m, gameMap)) {
                cout << "Invalid move: " << m.key << '\n';
                continue;
            }

            State newState = currState + m;
            string newSeq = actionSeq[currState.hashKey] + m.key;

            cout << "New State:\n";
            newState.debug();
            cout << newSeq << '\n';

            if (!visited.contain(newState)) {
                cout << "Not visited!\n";
                nodes.emplace_back(newState.keyNode);

                if (isSolution(newState, gameMap)) {
                    cout << "Solution!\n";
                    return newSeq;
                }
                else cout << "Not solution!\n";

                if (!isDeadlock(newState, gameMap)) {
                    Q.emplace(newState);
                    visited.insert(newState);
                    actionSeq[newState.hashKey] = newSeq;
                }
            }
            else delete newState.keyNode;
        }
    }
#endif

    return "";
}

void createMapData(char* &mapfile, GameMap& gameMap)
{
    std::ifstream testcase(mapfile);
    string line;

    if (testcase.is_open()) {
        while (std::getline(testcase, line)) {
            gameMap.tiles.emplace_back(line);
        }
        testcase.close();
    } 
    else 
        std::cerr << "Unable to open file.\n";

    for (int i = 0; i < gameMap.tiles.size(); i++) {
        for (int j = 0; j < gameMap.tiles[i].size(); j++) {
            char c = gameMap.tiles[i][j];

            if (c == 'x' || c == 'X')
                gameMap.boxPos.emplace_back(i, j);
            else if (c == 'o' || c == 'O')
                gameMap.playerPos = Coord(i, j);
            else if (c == '.' || c == 'X' || c == 'O')
                gameMap.targetPos.emplace_back(i, j);
        }
    }
}

int main(int argc, char* argv[])
{
    GameMap gameMap;
    vector<Node*> nodes;

    createMapData(argv[1], gameMap);

    std::cout << findActionSequence(gameMap, nodes);

    for (auto& n : nodes) {
        delete n;
    }

    return 0;
}
