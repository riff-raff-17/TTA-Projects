#include <SFML/Graphics.hpp>

const int WIDTH = 800, HEIGHT = 600;

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
        window.display();
    }
}