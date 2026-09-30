#include <SFML/Graphics.hpp>
#include <deque>
#include <cstdlib>
#include <ctime>

const int WIDTH = 800, HEIGHT = 600;
const int CELL = 20;
const float MOVE_INTERVAL = 0.12f;

sf::Vector2i randomFood(const std::deque<sf::Vector2i>& snake) {
    int cols = WIDTH / CELL, rows = HEIGHT / CELL;
    sf::Vector2i pos;
    do {
        pos = {std::rand() % cols, std::rand() % rows};
    } while ([&]{ for (auto& s : snake) if (s == pos) return true; return false; }());
    return pos;
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr))); // seed RNG

    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Snake");
    window.setFramerateLimit(60);

    // --- Snake ---
    std::deque<sf::Vector2i> snake;
    snake.push_back({WIDTH / CELL / 2, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 1, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 2, HEIGHT / CELL / 2});

    sf::Vector2i dir(1, 0);
    sf::Vector2i nextDir(1, 0);

    // --- Food ---
    sf::Vector2i food = randomFood(snake);

    // --- Timing ---
    float timer = 0.f;
    sf::Clock clock;

    // --- Reusable cell shape ---
    sf::RectangleShape cell(sf::Vector2f(CELL - 2.f, CELL - 2.f));

    while (window.isOpen()) {
        float dt = clock.restart().asSeconds();

        // --- Events ---
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                if (key->code == sf::Keyboard::Key::W     && dir.y == 0) nextDir = {0, -1};
                if (key->code == sf::Keyboard::Key::S     && dir.y == 0) nextDir = {0,  1};
                if (key->code == sf::Keyboard::Key::A     && dir.x == 0) nextDir = {-1, 0};
                if (key->code == sf::Keyboard::Key::D     && dir.x == 0) nextDir = {1,  0};
                if (key->code == sf::Keyboard::Key::Up    && dir.y == 0) nextDir = {0, -1};
                if (key->code == sf::Keyboard::Key::Down  && dir.y == 0) nextDir = {0,  1};
                if (key->code == sf::Keyboard::Key::Left  && dir.x == 0) nextDir = {-1, 0};
                if (key->code == sf::Keyboard::Key::Right && dir.x == 0) nextDir = {1,  0};
            }
        }

        // --- Update ---
        timer += dt;
        if (timer >= MOVE_INTERVAL) {
            timer = 0.f;
            dir = nextDir;

            sf::Vector2i head = snake.front() + dir;
            snake.push_front(head);

            if (head == food) {
                food = randomFood(snake); // eat: spawn new food, don't pop tail
            } else {
                snake.pop_back(); // normal move: remove tail
            }
        }

        // --- Draw ---
        window.clear(sf::Color(20, 20, 20));

        // Grid
        for (int x = 0; x < WIDTH; x += CELL) {
            sf::RectangleShape line(sf::Vector2f(1, HEIGHT));
            line.setPosition(sf::Vector2f(x, 0));
            line.setFillColor(sf::Color(40, 40, 40));
            window.draw(line);
        }
        for (int y = 0; y < HEIGHT; y += CELL) {
            sf::RectangleShape line(sf::Vector2f(WIDTH, 1));
            line.setPosition(sf::Vector2f(0, y));
            line.setFillColor(sf::Color(40, 40, 40));
            window.draw(line);
        }

        // Snake
        for (size_t i = 0; i < snake.size(); ++i) {
            float t = 1.f - (float)i / snake.size();
            cell.setFillColor(sf::Color(
                static_cast<uint8_t>(50 + 180 * t),
                static_cast<uint8_t>(200),
                static_cast<uint8_t>(50)
            ));
            cell.setPosition(sf::Vector2f(snake[i].x * CELL + 1.f, snake[i].y * CELL + 1.f));
            window.draw(cell);
        }

        // Food
        cell.setFillColor(sf::Color(220, 50, 50));
        cell.setPosition(sf::Vector2f(food.x * CELL + 1.f, food.y * CELL + 1.f));
        window.draw(cell);

        window.display();
    }
}
