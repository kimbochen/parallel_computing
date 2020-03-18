#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
    std::vector<Direction> dir = {
        { 'W', -1, 0 },
        { 'A', 0, -1 },
        { 'S', 1, 0 },
        { 'D', 0, 1 }
    };

    if (testcase.is_open()) {
        while (std::getline(testcase, line)) {
            game_map.emplace_back(line);
        }
        testcase.close();
    } else
        std::cerr << "Unable to open file.\n";

#ifdef DEBUG
    for (const auto& [key, dx, dy] : dir) {
        std::cerr << key << ' ' << dx << ' ' << dy << '\n';
    }
#endif

    return 0;
}
