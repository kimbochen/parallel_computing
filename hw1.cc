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

    friend Coord operator + (const Coord &a, const Coord &b)
    {
        return Coord(a.x + b.x, a.y + b.y);
    }

    friend Coord operator * (int k, const Coord &a)
    {
        return Coord(k * a.x, k * a.y);
    }

    friend bool operator == (const Coord &a, const Coord &b)
    {
        return (a.x == b.x && a.y == b.y);
    }

#ifdef DEBUG
    void debug() const
    {
        std::cout << '(' << x << ',' << y << ')';
    }
#endif

    int x, y;
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

struct MapData {
    vector<string> gameMap;
    vector<Coord> boxPos;
    Coord playerPos;
};

class State {
public:
    State() {}

    State(MapData &mapData, Node* prevNode = nullptr, char key = '\0')
    {
        gameMap = mapData.gameMap;
        playerPos = mapData.playerPos;
        boxPos = mapData.boxPos;
        keyNode = new Node(prevNode, key);
        computeHashKey();
    }

    State operator + (const Move& m) const
    {
        State state;
        Coord movedBoxPos = playerPos + 2 * m.dir;

        state.gameMap = gameMap;
        state.boxPos = boxPos;
        state.playerPos = playerPos + m.dir;
        state.keyNode = new Node(keyNode, m.key);

        char &prevPlayerTile = state.gameMap[playerPos.x][playerPos.y];
        prevPlayerTile = (prevPlayerTile == 'o') ? ' ' : '.';

        char &playerTile = state.gameMap[state.playerPos.x][state.playerPos.y];
        char &boxTile = state.gameMap[movedBoxPos.x][movedBoxPos.y];

        if (playerTile == ' ')
            playerTile = 'o';
        else if (playerTile == '.')
            playerTile = 'O';
        else {
            playerTile = (playerTile == 'x') ? 'o' : 'O';
            boxTile = (boxTile == ' ') ? 'x' : 'X';

            for (auto& bp : state.boxPos) {
                if (bp == state.playerPos) {
                    bp = bp + m.dir;
                    break;
                }
            }
        }

        state.computeHashKey();

        return state;
    }

    void computeHashKey()
    {
        std::ostringstream os;

        os << playerPos.x << playerPos.y;
        for (const auto& c : boxPos) {
            os << c.x << c.y;
        }
        hashKey = os.str();

    }

#ifdef DEBUG
    void debug() const 
    {
        using std::cout;

        cout << "Map:\n";
        for (const auto& s : gameMap) {
            cout << s << '\n';
        }

        cout << "boxPos: ";
        for (const auto& c : boxPos) {
            c.debug();
            cout << ' ';
        }

        cout << "  playerPos: ";
        playerPos.debug();

        cout << " keyNode: " << keyNode->key;

        cout << " hashKey: " << hashKey << '\n';
    }
#endif

    vector<string> gameMap;
    vector<Coord> boxPos;
    Coord playerPos;
    Node* keyNode;
    string hashKey;
};

class Record {
public:
    bool contain(const State &state)
    {
        auto it = history.find(state.hashKey);
        return (it != history.end());
    }

    void insert(const State &state)
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

bool isValidMove(const State& state, const Move& m)
{
    Coord newPlayerPos = state.playerPos + m.dir;
    char playerTile = state.gameMap[newPlayerPos.x][newPlayerPos.y];

    if (playerTile != '#') {
        if (playerTile == 'x' || playerTile == 'X') {
            Coord newBoxPos = state.playerPos + 2 * m.dir;
            char boxTile = state.gameMap[newBoxPos.x][newBoxPos.y];

            return (boxTile != '#');
        }
        else
            return true;
    }

    return false;
}

bool isSolution(const State& state)
{
    int n = state.gameMap.size() - 1;

    for (int i = 1; i < n; i++) {
        string s = state.gameMap[i];
        if (s.find_first_of(".xO") != string::npos) {
            return false;
        }
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

bool isDeadlock(const State& state)
{
    return false;
}

string findActionSequence(MapData &mapData, const vector<Coord> &targetPos)
{
    State initState = State(mapData);
    Record visited;
    queue<State> Q;
    vector<Node*> nodes;
    Move moves[4] = {
        { 'W', -1, 0 }, { 'A', 0, -1 }, { 'S', 1, 0 }, { 'D', 0, 1 }
    };
    std::unordered_map<std::string, std::string> actionSeq;

    Q.emplace(initState);
    visited.insert(initState);
    actionSeq[initState.hashKey] = "";

#ifndef ALGO
    while (!Q.empty()) {
        State currState = Q.front();
        Q.pop();

        for (int i = 0; i < 4; i++) {
            Move m = moves[i];

            if (!isValidMove(currState, m)) continue;

            State newState = currState + m;
            string newSeq = actionSeq[currState.hashKey] + m.key;

            if (!visited.contain(newState)) {
                visited.insert(newState);
                nodes.emplace_back(newState.keyNode);

                if (isSolution(newState)) {
                    for (auto& n : nodes) delete n;
                    return newSeq;
                }

                if (!isDeadlock(newState)) {
                    Q.emplace(newState);
                    actionSeq[newState.hashKey] = newSeq;
                }
            }
            else delete newState.keyNode;
        }
    }
#endif

#ifdef ALGO
    Q.emplace(initState);
    visited.insert(initState);
    actionSeq[initState.hashKey] = "";

    using std::cout;

    while (!Q.empty()) {
        State currState = Q.front();
        Q.pop();

        cout << "Current state:\n";
        currState.debug();
        cout << '\n';

        for (int i = 0; i < 4; i++) {
            Move m = moves[i];

            cout << "Current move: " << m.key << '\n';

            if (!isValidMove(currState, m)) {
                cout << "Invalid move: " << m.key << '\n';
                continue;
            }

            State newState = currState + m;
            string newSeq = actionSeq[currState.hashKey] + m.key;

            cout << "New State:\n";
            newState.debug();
            cout << "New sequence: " << newSeq << '\n';

            if (!visited.contain(newState)) {
                cout << "Not visited!\n";
                visited.insert(newState);
                nodes.emplace_back(newState.keyNode);

                if (isSolution(newState)) {
                    cout << "Solution!\n";
                    return newSeq;
                }
                else cout << "Not solution!\n";

                if (!isDeadlock(newState)) {
                    cout << "Not deadlock!\n";
                    Q.emplace(newState);
                    actionSeq[newState.hashKey] = newSeq;
                }
            }
            else {
                cout << "Visited\n";
                delete newState.keyNode;
            }
            cout << "Number of States in queue: " << Q.size() << '\n';
            cout << "\n\n\n";
        }
    }
#endif

    return "";
}

void createMapData(char* &mapfile, MapData &mapData, vector<Coord> &targetPos)
{
    std::ifstream testcase(mapfile);
    string line;

    if (testcase.is_open()) {
        while (std::getline(testcase, line)) {
            mapData.gameMap.emplace_back(line);
        }
        testcase.close();
    } 
    else 
        std::cerr << "Unable to open file.\n";

    for (int i = 0; i < mapData.gameMap.size(); i++) {
        for (int j = 0; j < mapData.gameMap[i].size(); j++) {
            char c = mapData.gameMap[i][j];

            if (c == 'x' || c == 'X')
                mapData.boxPos.emplace_back(i, j);
            if (c == '.' || c == 'X' || c == 'O')
                targetPos.emplace_back(i, j);
            if (c == 'o' || c == 'O')
                mapData.playerPos = Coord(i, j);
        }
    }
}

int main(int argc, char* argv[])
{
    MapData mapData;
    vector<Coord> targetPos;

    createMapData(argv[1], mapData, targetPos);

    std::cout << findActionSequence(mapData, targetPos) << '\n';

    return 0;
}
