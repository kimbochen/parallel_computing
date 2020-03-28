#include <boost/algorithm/string.hpp>
#include <boost/unordered_set.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#ifdef TBB
#include <tbb/concurrent_unordered_set.h>
#endif
#include <tuple>
#include <vector>

using Position = std::tuple<int, int>;
using Move = std::tuple<char, int, int>;
using Grid = std::vector<std::vector<char>>;
using Substate = std::tuple<int, int, std::string, int, int>;
using PosTable = boost::unordered_set<Position, boost::hash<Position>>; 

enum PositionIndex {X, Y};
enum SubstateIndex {SX, SY, SSEQ, SDX, SDY};
enum DeadlockIndex {W, A, S, D};
enum MoveIndex {KEY, DX, DY};

class Game {
public:
    Game(char* & filename)
    {
        std::ifstream inFile;
        std::stringstream ss;

        inFile.open(filename);
        ss << inFile.rdbuf();
        boost::split(initMap, ss.str(), boost::is_any_of("\n"));

        n = initMap.size() - 1;
        m = initMap[0].size() - 1;

        #pragma omp declare reduction (\
            merge : std::vector<Position> : \
            omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())\
        )

        #pragma omp parallel default(none) \
        shared(initMap, n, m, playerX, playerY, targetPos)
        {
            int i, j;

            #pragma omp for collapse(2) private(i, j) reduction(merge : targetPos)
            for (i = 0; i < n; i++) {
                for (j = 0; j < m; j++) {
                    if (initMap[i][j] == 'o' || initMap[i][j] == 'O') {
                        #pragma omp critical
                        {
                            playerX = i;
                            playerY = j;
                        }
                    }

                    if (initMap[i][j] == '.' || initMap[i][j] == 'O' || initMap[i][j] == 'X')
                        targetPos.emplace_back(i, j);
                }
            }
        }

        targetNum = targetPos.size();

        initH = heuristic(initMap);

        moves = {
            {'W', -1,  0},
            {'A',  0, -1},
            {'S',  1,  0},
            {'D',  0,  1}
        };
    }

    std::vector<Substate> getSubstates(const Grid& gameMap, int x, int y)
    {
        enum QueueIndex {QX, QY, SEQ};

        std::vector<Substate> substates;
        std::queue<std::tuple<int, int, std::string>> Q;
#ifdef TBB
        tbb::concurrent_unordered_set<Position, boost::hash<Position>> visited;
#else
        boost::unordered_set<Position, boost::hash<Position>> visited;
#endif

        Q.emplace(x, y, "");
        visited.insert({x, y});

        while (!Q.empty()) {
            int x = std::get<QX>(Q.front());
            int y = std::get<QY>(Q.front());
            std::string aseq = std::get<SEQ>(Q.front());
            Q.pop();

            for (int i = 0; i < 4; i++) {
                int dx = std::get<DX>(moves[i]);
                int dy = std::get<DY>(moves[i]);
                char key = std::get<KEY>(moves[i]);

                int nx = x + dx, ny = y + dy;
                char tile = gameMap[nx][ny];

                if ((tile == '.' || tile == ' ') && visited.find({nx, ny}) == visited.end()) {
                    visited.insert({nx, ny});
                    Q.emplace(nx, ny, aseq + key);
                }
                else if (tile == 'x' || tile == 'X') {
                    char boxTile = gameMap[nx + dx][ny + dy];

                    if (boxTile != 'x' && boxTile != 'X' && boxTile != '#')
                        substates.emplace_back(x, y, aseq + key, dx, dy);
                }
            }
        }

        return substates;
    }

    bool isSolved(const Grid &gmap)
    {
        bool solved = true;

        #pragma omp parallel default(none) shared(targetNum, solved, gmap)
        {
            int i, x, y;

            #pragma omp for private(i, x, y) reduction(&& : solved)
            for (i = 0; i < targetNum; i++) {
                x = std::get<X>(targetPos[i]);
                y = std::get<Y>(targetPos[i]);
                solved = solved && (gmap[x][y] == 'X');
            }
        }

        return solved;
    }

    int heuristic(const Grid &gmap)
    {
        std::vector<Position> boxPos;

        #pragma omp declare reduction (\
            merge : std::vector<Position> : \
            omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end())\
        )

        #pragma omp parallel default(none) shared(n, m, gmap, boxPos)
        {
            int i, j;
            #pragma omp for collapse(2) private(i, j) reduction(merge : boxPos)
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
                    if (gmap[i][j] == 'x')
                        boxPos.emplace_back(i, j);
                }
            }
        }

        int boxNum = boxPos.size();
        int h = 0;

        for (int i = 0; i < boxNum; i++) {
            int bx = std::get<0>(boxPos[i]);
            int by = std::get<1>(boxPos[i]);
            int dist = INT_MAX;

            #pragma omp parallel default(none) shared(i, targetNum, bx, by, dist, boxPos, targetPos)
            {
                int j;
                #pragma omp for private(j) reduction(min : dist)
                for (j = 0; j < targetNum; j++) {
                    int tx = std::get<0>(targetPos[i]);
                    int ty = std::get<1>(targetPos[i]);

                    dist = std::abs(bx - tx) + std::abs(by - ty);
                }
            }

            h += dist;
        }

        return -h;
    }

    bool isDeadlock(const Grid &gmap, PosTable &visited, int x, int y)
    {
        visited.insert({x, y});

        bool blocked[4];

        for (int i = 0; i < 4; i++) {
            int dx = std::get<DX>(moves[i]);
            int dy = std::get<DY>(moves[i]);
            int nx = x + dx, ny = y + dy;
            char c = gmap[x + dx][y + dy];

            if (c == 'x' || c == 'X') {
                if (visited.find({nx, ny}) == visited.end()) {
                    blocked[i] = isDeadlock(gmap, visited, nx, ny);
                }
            }
            else
                blocked[i] = (c == '#');
        }

        return ((blocked[W] || blocked[S]) && (blocked[A] || blocked[D]));
    }

    void solve()
    {
        using State = std::tuple<int, Grid, int, int, std::string>;
        enum StateIndex {H, GRID, PX, PY, SEQ};

        bool solved = false;
        std::priority_queue<State> stateQueue;
#ifdef TBB
        tbb::concurrent_unordered_set<Grid, boost::hash<Grid>> visited;
#else
        boost::unordered_set<Grid, boost::hash<Grid>> visited;
#endif

        stateQueue.emplace(initH, initMap, playerX, playerY, "");
        visited.insert(initMap);

        while (!stateQueue.empty() && !solved) {
            std::string actSeq = std::get<SEQ>(stateQueue.top());
            Grid gmap = std::get<GRID>(stateQueue.top());
            int h = std::get<H>(stateQueue.top());
            int x = std::get<PX>(stateQueue.top());
            int y = std::get<PY>(stateQueue.top());

            stateQueue.pop();

            std::vector<Substate> substates = getSubstates(gmap, x, y);
            int ssSize = substates.size();

            #pragma omp parallel default(none) \
            shared(std::cout, ssSize, substates, gmap, h, x, y, actSeq, visited, solved, stateQueue)
            {
                int i, px, py, dx, dy;
                int nx, ny, boxX, boxY;
                std::string newActSeq;
                Grid newGmap;

                #pragma omp for \
                private(i, px, py, dx, dy, nx, ny, boxX, boxY, newActSeq, newGmap)
                for (i = 0; i < ssSize; i++) {
                    px = std::get<SX>(substates[i]);
                    py = std::get<SY>(substates[i]);
                    dx = std::get<SDX>(substates[i]);
                    dy = std::get<SDY>(substates[i]);
                    newActSeq = std::get<SSEQ>(substates[i]);

                    newGmap = gmap;
                    nx = px + dx, ny = py + dy;
                    boxX = px + 2 * dx, boxY = py + 2 * dy;

                    newGmap[x][y] = (gmap[x][y] == 'o') ? ' ' : '.';
                    newGmap[nx][ny] = (gmap[nx][ny] == 'x') ? 'o' : 'O';
                    if (gmap[boxX][boxY] == '.' || gmap[boxX][boxY] == 'O')
                        newGmap[boxX][boxY] = 'X';
                    else
                        newGmap[boxX][boxY] = 'x';

                    if (isSolved(newGmap)) {
                        #pragma omp critical
                        {
                            std::cout << actSeq + newActSeq << '\n';
                        }
                        solved = true;
                    }

                    if (visited.find(newGmap) == visited.end()) {
                        boost::unordered_set<Position, boost::hash<Position>> v;

                        if (newGmap[boxX][boxY] == 'X' || !isDeadlock(newGmap, v, boxX, boxY)) {
                            #pragma omp critical
                            {
                                stateQueue.emplace(h + heuristic(newGmap), newGmap, nx, ny, actSeq + newActSeq);
                            }
                        }

                        visited.insert(newGmap);
                    }
                }
            }
        }
    }

private:
    Grid initMap;
    int n, m;
    int playerX, playerY;
    int initH;
    int targetNum;
    std::vector<Position> targetPos;
    std::vector<Move> moves;
};

int main(int argc, char* argv[])
{
    if (argc == 2) {
        Game game(argv[1]);
        game.solve();
    }
    else
        std::cerr << "Invalid number of arguments.\n";

    return 0;
}
