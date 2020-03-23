/*
 * initState <- (H, initGrid, playerX, playerY, "")
 * stateQueue <- initState
 *
 * while !stateQueue.empty() && !solved:
 *     currState <- stateQueue.pop()
 *     h, grid, x, y, seq = currState
 *
 *     for move in 4 possible moves:
 *          dx, dy, key = move
 *
 *          if canMakeMove(currState, x, y, dx, dy):
 *             newGrid = grid
 *             makeMove(newGrid, x, y, dx, dy)
 *          else:
 *              continue
 *
 *          isVisited = (visited.find(newGrid) != visited.end())
 *
 *          if !isVisited:
 *              visited.insert(newGrid)
 *
 *              if isSolved(newGrid):
 *                  ans = seq + key
 *                  solved = true
 *
 *              if !isDeadlock(newGrid, x+dx, y+dy):
 *                  newH = h + heuristic(newGrid, x+dx, y+dy)
 *                  stateQueue <- (newH, newGrid, x+dx, y+dy)
 */

#include <boost/algorithm/string.hpp>
#include <boost/unordered_set.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using GameMap = std::vector<std::vector<char>>;
using State = std::tuple<int, GameMap, int, int, std::string>;

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

        // Initialize h
        initH = heuristic(initField);

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

    int heuristic(const GameMap &field)
    {
        int n = field.size() - 1;
        int m = field[0].size() - 1;
        int i, j;
        std::vector<std::tuple<int, int>> boxPos;
        std::vector<std::tuple<int, int>> tarPos;

        #pragma omp declare reduction (\
            merge : std::vector<std::tuple<int, int>> : \
            omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())\
        )

        #pragma omp parallel default(none) shared(i, j, n, m, field, boxPos, tarPos)
        {
            #pragma omp for collapse(2) reduction(merge : boxPos) reduction(merge : tarPos)
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
                    if (field[i][j] == 'x')
                            boxPos.emplace_back(i, j);
                    else if (field[i][j] == '.' || field[i][j] == 'O')
                            tarPos.emplace_back(i, j);
                }
            }
        }

        int b = boxPos.size();
        int t = tarPos.size();
        int h = 0;

        for (i = 0; i < b; i++) {
            int bx = std::get<0>(boxPos[i]);
            int by = std::get<1>(boxPos[i]);
            int dist = INT_MAX;

            #pragma omp parallel default(none) shared(i, j, t, bx, by, dist, boxPos, tarPos)
            {
                #pragma omp for private(j) reduction(min : dist)
                for (j = 0; j < t; j++) {
                    int tx = std::get<0>(tarPos[i]);
                    int ty = std::get<1>(tarPos[i]);

                    dist = std::abs(bx - tx) + std::abs(by - ty);
                }
            }

            h += dist;
        }


        return -h;
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

    bool isDeadlock(const GameMap &field, int x, int y, int dx, int dy)
    {
        int boxX = x + dx, boxY = y + dy;

        if (field[boxX][boxY] != 'x') return false;

        std::string adjTiles(6, '~');

        for (int i = 0; i < 4; i++) {
            int adjX = boxX + std::get<1>(direction[i]);
            int adjY = boxY + std::get<2>(direction[i]);
            adjTiles[i + 1] = field[adjX][adjY];
        }
        adjTiles[5] = adjTiles[1];

        return (adjTiles.find("##") != std::string::npos);
    }

    std::string solve()
    {
        boost::unordered_set<GameMap, boost::hash<GameMap>> visited;
        std::priority_queue<State> stateQueue;
        bool foundSol = false;

        stateQueue.emplace(initH, initField, playerPosX, playerPosY, "");
        visited.insert(initField);

#ifdef DEBUG
        int iterNum = 0;
#endif
        while(!stateQueue.empty() && !foundSol) {
#ifdef DEBUG
            iterNum++;
#endif
            State currState = stateQueue.top();
            stateQueue.pop();

            int h = std::get<0>(currState);
            GameMap field = std::get<1>(currState);
            int x = std::get<2>(currState);
            int y = std::get<3>(currState);
            std::string actionSeq = std::get<4>(currState);

            #pragma omp parallel default(none) \
            shared(ans, actionSeq, direction, field, x, y, visited, stateQueue, foundSol, h)
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

                        if (!isDeadlock(newField, x + dx, y + dy, dx, dy)) {
                            #pragma omp critical
                            {
                                int newH = h + heuristic(newField);
                                stateQueue.emplace(newH, newField, x + dx, y + dy, actionSeq + key);
                            }
                        }
                    }
                }
            }
        }

#ifdef DEBUG
        std::cout << "iterNum: " << iterNum << '\n';
#endif

        return ans;
    }

private:
    std::string ans;
    GameMap initField;
    int playerPosX, playerPosY, initH;
    std::vector<std::tuple<char, int, int>> direction;
};

int main(int argc, char* argv[])
{
    Game game(argv[1]);

    std::cout << game.solve() << '\n';

	return 0;
}
