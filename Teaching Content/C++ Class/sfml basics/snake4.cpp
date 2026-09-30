#include <SFML/Graphics.hpp>
#include <deque>

const int WIDTH = 800, HEIGHT = 600;
const int CELL = 20;
const float MOVE_INTERVAL = 0.12f; // seconds between snake moves

int main(){
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Snake");
    window.setFramerateLimit(60);

    // --- Snake ---
    std::deque<sf::Vector2i> snake;
    snake.push_back({WIDTH / CELL / 2, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 1, HEIGHT / CELL / 2});
    snake.push_back({WIDTH / CELL / 2 - 2, HEIGHT / CELL / 2});

    sf::Vector2i dir(1, 0);  // moving right

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
        }

        // --- Update ---
        timer += dt;
        if (timer >= MOVE_INTERVAL) {
            timer = 0.f;

            sf::Vector2i head = snake.front() + dir;
            snake.push_front(head);
            snake.pop_back();
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
        for (int y = 0; y < HEIGHT; y += CELL){
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

        window.display();
    }
}