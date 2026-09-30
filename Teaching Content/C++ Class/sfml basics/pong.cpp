#include <SFML/Graphics.hpp>
#include <string>

const int WIDTH = 800, HEIGHT = 600;
const float PADDLE_SPEED = 400.f;
const float BALL_SPEED = 350.f;

int main() {
    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Pong");
    window.setFramerateLimit(60);

    // --- Paddles ---
    sf::RectangleShape leftPaddle({15, 100});
    sf::RectangleShape rightPaddle({15, 100});
    leftPaddle.setPosition({20, HEIGHT / 2.f - 50});
    rightPaddle.setPosition({WIDTH - 35.f, HEIGHT / 2.f - 50});

    // --- Ball ---
    sf::CircleShape ball(10);
    ball.setPosition({WIDTH / 2.f - 10, HEIGHT / 2.f - 10});
    sf::Vector2f ballVel(BALL_SPEED, BALL_SPEED);

    // --- Font & Score ---
    sf::Font font;
    if (!font.openFromFile("/Users/rafalr/Library/Fonts/FiraCodeNerdFont-Light.ttf")) {
        return -1; // exit if font fails to load
    }

    sf::Text scoreText(font);
    scoreText.setCharacterSize(36);
    scoreText.setPosition({WIDTH / 2.f - 40, 10});

    int leftScore = 0, rightScore = 0;
    sf::Clock clock;
    // ... rest of your code

    while (window.isOpen()) {
        float dt = clock.restart().asSeconds();

        // --- Events (SFML 3.0 style) ---
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
        }

        // --- Left Paddle (W/S) ---
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W) && leftPaddle.getPosition().y > 0)
            leftPaddle.move({0, -PADDLE_SPEED * dt});
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S) &&
            leftPaddle.getPosition().y + 100 < HEIGHT)
            leftPaddle.move({0, PADDLE_SPEED * dt});

        // --- Right Paddle (Up/Down) ---
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up) && rightPaddle.getPosition().y > 0)
            rightPaddle.move({0, -PADDLE_SPEED * dt});
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down) &&
            rightPaddle.getPosition().y + 100 < HEIGHT)
            rightPaddle.move({0, PADDLE_SPEED * dt});

        // --- Ball Movement ---
        ball.move(ballVel * dt);
        auto bpos = ball.getPosition();

        // Top/bottom bounce
        if (bpos.y <= 0 || bpos.y + 20 >= HEIGHT)
            ballVel.y = -ballVel.y;

        // Paddle collisions
        if (ball.getGlobalBounds().findIntersection(leftPaddle.getGlobalBounds()) ||
            ball.getGlobalBounds().findIntersection(rightPaddle.getGlobalBounds()))
            ballVel.x = -ballVel.x;

        // --- Scoring ---
        if (bpos.x < 0) {
            rightScore++;
            ball.setPosition({WIDTH / 2.f - 10, HEIGHT / 2.f - 10});
            ballVel = {BALL_SPEED, BALL_SPEED};
        }
        if (bpos.x > WIDTH) {
            leftScore++;
            ball.setPosition({WIDTH / 2.f - 10, HEIGHT / 2.f - 10});
            ballVel = {-BALL_SPEED, BALL_SPEED};
        }

        scoreText.setString(std::to_string(leftScore) + "  " + std::to_string(rightScore));

        // --- Draw ---
        window.clear(sf::Color::Black);
        window.draw(leftPaddle);
        window.draw(rightPaddle);
        window.draw(ball);
        window.draw(scoreText);
        window.display();
    }
}