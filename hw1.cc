#include <boost/algorithm/string.hpp>
#include <boost/unordered_set.hpp>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using GameMap = std::vector<std::vector<char>>;
using State = std::tuple<GameMap, int, int, std::string>;

class Game {
public:
    Game(char* &filename)
    {
        // Read file
        std::ifstream inFile;
        std::stringstream ss;

        inFile.open(filename);
        ss << inFile.rdbuf();
        boost::split(initField, ss.str(), boost::is_any_of("\n"));

        // Find player position
        int n = initField.size() - 1;
        int m = initField[0].size() - 1;

        #pragma omp parallel default(none) shared(initField, n, m, playerPosX, playerPosY)
        {
            int i, j;

            #pragma omp for collapse(2) private(i, j)
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
                    if (initField[i][j] == 'o' || initField[i][j] == 'O') {
                        #pragma omp critical
                        {
                            playerPosX = i;
                            playerPosY = j;
                        }
                    }
                }
            }
        }

        // Initialize directions
        direction = {
            {'W', -1,  0},
            {'A',  0, -1},
            {'S',  1,  0},
            {'D',  0,  1}
        };

        // Initialize answer string
        ans = "Not Solved!\n";
    }

    bool isValidStep(const GameMap &field, int x, int y, int dx, int dy)
    {
        char C2 = field[x + dx][y + dy];

        if (C2 == ' ' || C2 == '.') {
            return true;
        }
        else if (C2 == 'x' || C2 == 'X') {
            char C3 = field[x + 2 * dx][y + 2 * dy];
            return (C3 == ' ' || C3 == '.');
        }
        else return false;
    }

    void makeStep(GameMap &field, int x, int y, int dx, int dy)
    {
        field[x][y] = (field[x][y] == 'o') ? ' ' : '.';

        char &C2 = field[x + dx][y + dy];

        if (C2 == ' ') {
            C2 = 'o';
        }
        else if (C2 == '.') {
            C2 = 'O';
        }
        else {
            char &C3 = field[x + 2 * dx][y + 2 * dy];

            C3 = (C3 == ' ') ? 'x' : 'X';
            C2 = (C2 == 'x') ? 'o' : 'O';
        }
    }

    bool isSolution(const GameMap &field)
    {
        int n = field.size() - 1;
        int m = field[0].size() - 1;
        bool isSol = true;

        #pragma omp parallel default(none) shared(n, m, isSol, field)
        {
            int i, j;
            char c;

            #pragma omp for private(i, j, c) reduction(&& : isSol)
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
                    c = field[i][j];
                    isSol = isSol && (c != 'O' && c != 'x' && c != '.');
                }
            }
        }

        return isSol;
    }

    bool isDeadlock(const GameMap &field, int x, int y)
    {
        int n = field.size() - 1;
        int m = field[0].size() - 1;

        // A box is in a corner:
        // ##~~ -> T
        // ~##~ -> T
        // ~~## -> T
        // #~~# -> T
        bool isCorner = false;

        #pragma omp parallel default(none) shared(n, m, field, isCorner)
        {
            int i, j, k;
            int dx1, dy1, dx2, dy2;
            char adj1, adj2;

            #pragma omp for private(i, j, k) reduction(|| : isCorner)
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
                    if (field[i][j] == 'x') {
                        for (k = 0; k < 4; k++) {
                            dx1 = std::get<1>(direction[k]);
                            dy1 = std::get<2>(direction[k]);
                            adj1 = field[i + dx1][j + dy1];

                            dx2 = std::get<1>(direction[(k + 1) % 4]);
                            dy2 = std::get<2>(direction[(k + 1) % 4]);
                            adj2 = field[i + dx2][j + dy2];

                            isCorner = isCorner || (adj1 == '#' && adj2 == '#');
                        }
                    }
                }
            }
        }

        if (isCorner) return true;

        return false;
    }

    std::string solve()
    {
        boost::unordered_set<GameMap, boost::hash<GameMap>> visited;
        std::queue<State> stateQueue;
        bool foundSol = false;

        stateQueue.emplace(initField, playerPosX, playerPosY, "");
        visited.insert(initField);

        while(!stateQueue.empty() && !foundSol) {
            State currState = stateQueue.front();
            stateQueue.pop();

            GameMap field = std::get<0>(currState);
            int x = std::get<1>(currState);
            int y = std::get<2>(currState);
            std::string actionSeq = std::get<3>(currState);

#ifdef DEBUG
            std::cout << "Field:\n";
            for (auto &r : field) {
                for (auto &c : r) std::cout << c;
                std::cout << '\n';
            }
            std::cout << "PlayerPos: " << x << ' ' << y << ' ';
            std::cout << "actionSeq: " << actionSeq << '\n';
#endif

            #pragma omp parallel default(none) \
            shared(ans, actionSeq, direction, field, x, y, visited, stateQueue, foundSol)
            {
                int i, dx, dy;
                char key;
                GameMap newField;
                bool isVisited;

                #pragma omp for private(i, key, dx, dy, newField, isVisited)
                for (i = 0; i < 4; i++) {
                    key = std::get<0>(direction[i]);
                    dx = std::get<1>(direction[i]);
                    dy = std::get<2>(direction[i]);

                    if (!isValidStep(field, x, y, dx, dy)) continue;

                    newField = field;

                    makeStep(newField, x, y, dx, dy);

                    #pragma omp critical
                    {
                        isVisited = (visited.find(newField) != visited.end());
                        if (!isVisited) visited.insert(newField);
                    }

                    if (!isVisited) {
                        if (isSolution(newField)) {
                            #pragma omp critical
                            {
                                ans = actionSeq + key;
                                foundSol = true;
                            }
                        }

                        if (!isDeadlock(newField, x + dx, y + dy)) {
                            #pragma omp critical
                            {
                                stateQueue.emplace(newField, x + dx, y + dy, actionSeq + key);
                            }
                        }
                    }
                }
            }
        }

        return ans;
    }

#ifdef DEBUG
    void debug()
    {
        using std::cout;

        cout << "initField:\n";
        for (const auto &r : initField) {
            for (const auto &c : r) {
                cout << c;
            }
            cout << '\n';
        }

        cout << "Direction: ";
        for (const auto &d : direction) {
            cout << std::get<0>(d) << ',' << std::get<1>(d) << ','<< std::get<2>(d);
            cout << "   ";
        }
        cout << '\n';

        cout << "playerPos: " << playerPosX << ',' << playerPosY << '\n';
        cout << "isSolution: " << isSolution(initField) << '\n';
        cout << "isDeadlock: " << isDeadlock(initField, playerPosX, playerPosY) << '\n';

        for (int i = 0; i < 4; i++) {
            int dx = std::get<1>(direction[i]);
            int dy = std::get<2>(direction[i]);

            if (isValidStep(initField, playerPosX, playerPosY, dx, dy)) {
                GameMap field = initField;
                makeStep(field, playerPosX, playerPosY, dx, dy);
                cout << "newField:\n";
                for (const auto &r : field) {
                    for (const auto &c : r) {
                        cout << c;
                    }
                    cout << '\n';
                }
                cout << "isSolution: " << isSolution(field) << '\n';
                cout << "isDeadlock: " << isDeadlock(field, playerPosX, playerPosY) << '\n';
            }
        }

        cout << "\n\n\n";
    }
#endif

private:
    std::string ans;
    GameMap initField;
    int playerPosX, playerPosY;
    std::vector<std::tuple<char, int, int>> direction;
};

int main(int argc, char* argv[])
{
    Game game(argv[1]);

    std::cout << game.solve() << '\n';

	return 0;
}
