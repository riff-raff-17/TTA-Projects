#include <SFML/Graphics.hpp>
#include <deque>
#include <cstdlib>
#include <string>

const int WIDTH = 800, HEIGHT = 600;
const int CELL = 20; // grid cell size in pixels
const float MOVE_INTERVAL = 0.12f; // seconds between snake steps

sf::Vector2i randomFood(const std::deque<sf::Vector2i>& snake) {
    int cols = WIDTH / CELL, rows = HEIGHT / CELL;
    sf::Vector2i pos;
    do {
        pos = {std::rand() % cols, std::rand() % rows};
    } while ([&]{ for (auto& s : snake) if (s == pos) return true; return false; }());
    return pos;
}

int main() {
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Snake");
    window.setFramerateLimit(60);

    // --- Font & Score ---
    sf::Font font;
    if (!font.openFromFile("/Users/rafalr/Library/Fonts/FiraCodeNerdFont-Light.ttf"))
        return -1;

    sf::Text scoreText(font);
    scoreText.setCharacterSize(24);
    scoreText.setPosition({10, 10});

    sf::Text msgText(font);
    msgText.setCharacterSize(36);
    msgText.setFillColor(sf::Color::Yellow);

    // --- Snake ---
    std::deque<sf::Vector2i> snake;
    snake.push_back({WIDTH / CELL / 2, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 1, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 2, HEIGHT / CELL / 2});

    sf::Vector2i dir(1, 0);     // moving right
    sf::Vector2i nextDir(1, 0);

    // --- Food ---
    sf::Vector2i food = randomFood(snake);

    // --- State ---
    int score = 0;
    bool gameOver = false;
    float timer = 0.f;
    sf::Clock clock;

    // --- Reusable rectangle for drawing cells ---
    sf::RectangleShape cell(sf::Vector2f(CELL - 2.f, CELL - 2.f)); // 1px gap

    auto resetGame = [&]() {
        snake.clear();
        snake.push_back({WIDTH / CELL / 2, HEIGHT / CELL / 2});
        snake.push_back({WIDTH / CELL / 2 - 1, HEIGHT / CELL / 2});
        snake.push_back({WIDTH / CELL / 2 - 2, HEIGHT / CELL / 2});
        dir = {1, 0};
        nextDir = {1, 0};
        food = randomFood(snake);
        score = 0;
        gameOver = false;
        timer = 0.f;
    };

    while (window.isOpen()) {
        float dt = clock.restart().asSeconds();

        // --- Events ---
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();

            if (const auto* key = event->getIf<sf::Event::KeyPressed>()) {
                // Direction input — prevent 180-degree reversal
                if (key->code == sf::Keyboard::Key::W && dir.y == 0) nextDir = {0, -1};
                if (key->code == sf::Keyboard::Key::S && dir.y == 0) nextDir = {0,  1};
                if (key->code == sf::Keyboard::Key::A && dir.x == 0) nextDir = {-1, 0};
                if (key->code == sf::Keyboard::Key::D && dir.x == 0) nextDir = {1,  0};
                if (key->code == sf::Keyboard::Key::Up    && dir.y == 0) nextDir = {0, -1};
                if (key->code == sf::Keyboard::Key::Down  && dir.y == 0) nextDir = {0,  1};
                if (key->code == sf::Keyboard::Key::Left  && dir.x == 0) nextDir = {-1, 0};
                if (key->code == sf::Keyboard::Key::Right && dir.x == 0) nextDir = {1,  0};

                // Restart on R after game over
                if (key->code == sf::Keyboard::Key::R && gameOver)
                    resetGame();
            }
        }

        if (!gameOver) {
            timer += dt;
            if (timer >= MOVE_INTERVAL) {
                timer = 0.f;
                dir = nextDir;

                sf::Vector2i head = snake.front() + dir;

                // Wall collision
                if (head.x < 0 || head.x >= WIDTH / CELL ||
                    head.y < 0 || head.y >= HEIGHT / CELL)
                    gameOver = true;

                // Self collision
                for (auto& s : snake)
                    if (s == head) { gameOver = true; break; }

                if (!gameOver) {
                    snake.push_front(head);

                    if (head == food) {
                        score++;
                        food = randomFood(snake);
                        // Grow: don't pop the tail
                    } else {
                        snake.pop_back();
                    }
                }
            }
        }

        // --- Draw ---
        window.clear(sf::Color(20, 20, 20));

        // Draw faint grid
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

        // Draw food
        cell.setFillColor(sf::Color(220, 50, 50));
        cell.setPosition(sf::Vector2f(food.x * CELL + 1.f, food.y * CELL + 1.f));
        window.draw(cell);

        // Draw snake
        for (size_t i = 0; i < snake.size(); ++i) {
            float t = 1.f - (float)i / snake.size();  // gradient head→tail
            cell.setFillColor(sf::Color(
                static_cast<uint8_t>(50 + 180 * t),
                static_cast<uint8_t>(200),
                static_cast<uint8_t>(50)
            ));
            cell.setPosition(sf::Vector2f(snake[i].x * CELL + 1.f, snake[i].y * CELL + 1.f));
            window.draw(cell);
        }

        // Score
        scoreText.setString("Score: " + std::to_string(score));
        window.draw(scoreText);

        // Game over overlay
        if (gameOver) {
            msgText.setString("GAME OVER  |  R to restart");
            auto bounds = msgText.getLocalBounds();
            msgText.setPosition({
                WIDTH / 2.f - bounds.size.x / 2.f,
                HEIGHT / 2.f - bounds.size.y / 2.f
            });
            window.draw(msgText);
        }

        window.display();
    }
}