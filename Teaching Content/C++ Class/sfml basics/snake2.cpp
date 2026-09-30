#include <SFML/Graphics.hpp>

const int WIDTH = 800, HEIGHT = 600;
const int CELL = 20;  // grid cell size in pixels

int main() {
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Snake");
    window.setFramerateLimit(60);

    while (window.isOpen()) {
        // --- Events ---
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        // --- Draw ---
        window.clear(sf::Color(20, 20, 20));

        // Draw faint vertical lines
        for (int x = 0; x < WIDTH; x += CELL) {
            sf::RectangleShape line(sf::Vector2f(1, HEIGHT));
            line.setPosition(sf::Vector2f(x, 0));
            line.setFillColor(sf::Color(40, 40, 40));
            window.draw(line);
        }

        // Draw faint horizontal lines
        for (int y = 0; y < HEIGHT; y += CELL) {
            sf::RectangleShape line(sf::Vector2f(WIDTH, 1));
            line.setPosition(sf::Vector2f(0, y));
            line.setFillColor(sf::Color(40, 40, 40));
            window.draw(line);
        }

        window.display();
    }
}