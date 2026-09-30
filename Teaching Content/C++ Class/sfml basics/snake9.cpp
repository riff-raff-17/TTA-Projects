#include <SFML/Graphics.hpp>
#include <deque>
#include <cstdlib>
#include <string>

// --- Constants ---
const int WIDTH = 800, HEIGHT = 600;

// -----------------------------------------------------------------------
// NEW: Config - replaces the old compile-time constants CELL/MOVE_INTERVAL
// -----------------------------------------------------------------------
struct Config
{
    int cell = 20;
    float moveInterval = 0.12f;
    bool wrapWalls = false;
    // Snake color: head and tail colors for the gradient
    uint8_t r = 50, g = 200, b = 50;
    uint8_t tr = 10, tg = 60, tb = 10; // tail color
};

// -----------------------------------------------------------------------
// NEW: Game state machine
// -----------------------------------------------------------------------
enum class GameState
{
    Menu,
    Playing,
    GameOver
};

// -----------------------------------------------------------------------
// randomFood — unchanged, but now takes cell count from config
// -----------------------------------------------------------------------
sf::Vector2i randomFood(const std::deque<sf::Vector2i> &snake, const Config &cfg)
{
    int cols = WIDTH / cfg.cell, rows = HEIGHT / cfg.cell;
    sf::Vector2i pos;
    do
    {
        pos = {std::rand() % cols, std::rand() % rows};
    } while ([&]
             { for (auto& s : snake) if (s == pos) return true; return false; }());
    return pos;
}

// -----------------------------------------------------------------------
// NEW: Simple button helper
// -----------------------------------------------------------------------
struct Button
{
    sf::RectangleShape box;
    sf::Text label;

    Button(const sf::Font &font, const std::string &text,
           sf::Vector2f pos, sf::Vector2f size, unsigned charSize = 22)
        : label(font)
    {
        box.setSize(size);
        box.setPosition(pos);
        box.setFillColor(sf::Color(60, 60, 60));
        box.setOutlineThickness(2.f);
        box.setOutlineColor(sf::Color(100, 100, 100));

        label.setString(text);
        label.setCharacterSize(charSize);
        // Center text inside the box
        auto b = label.getLocalBounds();
        label.setPosition({pos.x + size.x / 2.f - b.size.x / 2.f,
                           pos.y + size.y / 2.f - b.size.y / 2.f - 2.f});
    }

    // Highlight when selected
    void setSelected(bool selected)
    {
        box.setFillColor(selected ? sf::Color(80, 130, 80) : sf::Color(60, 60, 60));
        box.setOutlineColor(selected ? sf::Color::Green : sf::Color(100, 100, 100));
    }

    // Hover effect
    void setHovered(bool hovered)
    {
        if (hovered)
            box.setOutlineColor(sf::Color::White);
    }

    bool contains(sf::Vector2f point) const
    {
        return box.getGlobalBounds().contains(point);
    }

    void draw(sf::RenderWindow &window) const
    {
        window.draw(box);
        window.draw(label);
    }
};

// --- Main Loop ---
int main()
{
    std::srand(static_cast<unsigned>(std::time(nullptr))); // seed RNG
    // std::srand(1); // set seed

    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Snake");
    window.setFramerateLimit(60);

    // --- Font & Text ---
    sf::Font font;
    if (!font.openFromFile("FiraCodeNerdFont-Regular.ttf"))
        return -1;

    // -----------------------------------------------------------------------
    // NEW: Menu buttons
    // Each row has mutually exclusive options; clicking one selects it and deselects the other
    // -----------------------------------------------------------------------
    // Speed row
    std::vector<Button> speedBtns = {
        Button(font, "Slow", {160.f, 160.f}, {120.f, 44.f}),
        Button(font, "Normal", {300.f, 160.f}, {120.f, 44.f}),
        Button(font, "Fast", {440.f, 160.f}, {120.f, 44.f}),
    };
    int speedSel = 1; // default: Normal

    // Grid size row
    std::vector<Button> gridBtns = {
        Button(font, "Small", {160.f, 260.f}, {120.f, 44.f}),
        Button(font, "Medium", {300.f, 260.f}, {120.f, 44.f}),
        Button(font, "Large", {440.f, 260.f}, {120.f, 44.f}),
    };
    int gridSel = 1; // default: Medium

    // Color row
    std::vector<Button> colorBtns = {
        Button(font, "Green", {160.f, 360.f}, {120.f, 44.f}),
        Button(font, "Blue", {300.f, 360.f}, {120.f, 44.f}),
        Button(font, "Fire", {440.f, 360.f}, {120.f, 44.f}),
    };
    int colorSel = 0; // default: Green

    // Wall mode row
    std::vector<Button> wallBtns = {
        Button(font, "Hard", {160.f, 460.f}, {120.f, 44.f}),
        Button(font, "Wrap", {300.f, 460.f}, {120.f, 44.f}),
    };
    int wallSel = 0; // default: Hard

    // Play button
    Button playBtn(font, "PLAY", {330.f, 530.f}, {140.f, 50.f}, 28);

    // Menu labels
    auto makeLabel = [&](const std::string &text, float y)
    {
        sf::Text t(font);
        t.setString(text);
        t.setCharacterSize(20);
        t.setFillColor(sf::Color(180, 180, 180));
        t.setPosition({60.f, y + 12.f});
        return t;
    };
    sf::Text titleText(font);
    titleText.setString("SNAKE");
    titleText.setCharacterSize(56);
    titleText.setFillColor(sf::Color::Green);
    {
        auto b = titleText.getLocalBounds();
        titleText.setPosition({WIDTH / 2.f - b.size.x / 2.f, 60.f});
    };

    auto labelSpeed = makeLabel("Speed", 160.f);
    auto labelGrid = makeLabel("Grid", 260.f);
    auto labelColor = makeLabel("Color", 360.f);
    auto labelWalls = makeLabel("Walls", 460.f);

    // -----------------------------------------------------------------------
    // Game variables — same as step 8, now driven by Config
    // -----------------------------------------------------------------------
    Config cfg;
    GameState state = GameState::Menu;

    std::deque<sf::Vector2i> snake;
    sf::Vector2i dir(1, 0), nextDir(1, 0);
    sf::Vector2i food;
    int score = 0;
    float timer = 0.f;
    sf::Clock clock;

    sf::Text scoreText(font);
    scoreText.setCharacterSize(24);
    scoreText.setFillColor(sf::Color::White);
    scoreText.setPosition({10.f, 10.f});

    sf::Text msgText(font);
    msgText.setCharacterSize(36);
    msgText.setFillColor(sf::Color::Yellow);

    sf::RectangleShape cell; // size set when game starts

    // Apply config and initialize/reset game state
    auto startGame = [&]()
    {
        // Map selections to config values
        float speeds[] = {0.22f, 0.12f, 0.06f};
        int cells[] = {28, 20, 14};
        cfg.moveInterval = speeds[speedSel];
        cfg.cell = cells[gridSel];
        cfg.wrapWalls = (wallSel == 1);

        // Color themes: head (r, g, b) and tail (tr, tg, tb)
        uint8_t rs[] = {50, 50, 230};  // head R  (green, blue, fire)
        uint8_t gs[] = {200, 150, 80}; // head G
        uint8_t bs[] = {50, 230, 50};  // head B
        uint8_t trs[] = {10, 10, 80};  // tail R  (dark green, dark navy, dark maroon)
        uint8_t tgs[] = {60, 30, 15};  // tail G
        uint8_t tbs[] = {10, 80, 10};  // tail B
        cfg.r = rs[colorSel];
        cfg.g = gs[colorSel];
        cfg.b = bs[colorSel];
        cfg.tr = trs[colorSel];
        cfg.tg = tgs[colorSel];
        cfg.tb = tbs[colorSel];

        cell.setSize(sf::Vector2f(cfg.cell - 2.f, cfg.cell - 2.f));

        int midX = (WIDTH / cfg.cell) / 2;
        int midY = (HEIGHT / cfg.cell) / 2;
        snake.clear();
        snake.push_back({midX, midY});
        snake.push_back({midX - 1, midY});
        snake.push_back({midX - 2, midY});
        dir = nextDir = {1, 0};
        food = randomFood(snake, cfg);
        score = 0;
        timer = 0.f;
        state = GameState::Playing;
    };

    auto resetGame = [&]()
    {
        state = GameState::Menu;
    };

    // -----------------------------------------------------------------------
    // Main loop - unchanged structure; behavior switches on 'state'
    // -----------------------------------------------------------------------
    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();
        sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

        // --- Events ---
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();

            // --- MENU input ---
            if (state == GameState::Menu)
            {
                if (const auto *mb = event->getIf<sf::Event::MouseButtonPressed>())
                {
                    if (mb->button == sf::Mouse::Button::Left)
                    {
                        // Check each button group
                        for (int i = 0; i < (int)speedBtns.size(); i++)
                            if (speedBtns[i].contains(mousePos))
                                speedSel = i;
                        for (int i = 0; i < (int)gridBtns.size(); i++)
                            if (gridBtns[i].contains(mousePos))
                                gridSel = i;
                        for (int i = 0; i < (int)colorBtns.size(); i++)
                            if (colorBtns[i].contains(mousePos))
                                colorSel = i;
                        for (int i = 0; i < (int)wallBtns.size(); i++)
                            if (wallBtns[i].contains(mousePos))
                                wallSel = i;
                        if (playBtn.contains(mousePos))
                            startGame();
                    }
                }
            }

            // --- PLAYING / GAME OVER input ---
            if (state == GameState::Playing || state == GameState::GameOver)
            {
                if (const auto *key = event->getIf<sf::Event::KeyPressed>())
                {
                    if (state == GameState::Playing)
                    {
                        if (key->code == sf::Keyboard::Key::W && dir.y == 0)
                            nextDir = {0, -1};
                        if (key->code == sf::Keyboard::Key::S && dir.y == 0)
                            nextDir = {0, 1};
                        if (key->code == sf::Keyboard::Key::A && dir.x == 0)
                            nextDir = {-1, 0};
                        if (key->code == sf::Keyboard::Key::D && dir.x == 0)
                            nextDir = {1, 0};
                        if (key->code == sf::Keyboard::Key::Up && dir.y == 0)
                            nextDir = {0, -1};
                        if (key->code == sf::Keyboard::Key::Down && dir.y == 0)
                            nextDir = {0, 1};
                        if (key->code == sf::Keyboard::Key::Left && dir.x == 0)
                            nextDir = {-1, 0};
                        if (key->code == sf::Keyboard::Key::Right && dir.x == 0)
                            nextDir = {1, 0};
                    }
                    if (key->code == sf::Keyboard::Key::R && state == GameState::GameOver)
                        resetGame();
                }
            }
        }

        // --- Update (Playing only) ---
        if (state == GameState::Playing)
        {
            timer += dt;
            if (timer >= cfg.moveInterval)
            {
                timer = 0.f;
                dir = nextDir; // commit the queued dir only on each step

                sf::Vector2i head = snake.front() + dir;
                int cols = WIDTH / cfg.cell, rows = HEIGHT / cfg.cell;

                if (cfg.wrapWalls)
                {
                    // wrap-around - modulo teleports to opposite side
                    head.x = (head.x + cols) % cols;
                    head.y = (head.y + rows) % rows;
                }
                else
                {
                    // Hard walls
                    if (head.x < 0 || head.x >= cols ||
                        head.y < 0 || head.y >= rows)
                        state = GameState::GameOver;
                }

                // Self collision
                for (auto &s : snake)
                    if (s == head)
                    {
                        state = GameState::GameOver;
                        break;
                    }

                if (state == GameState::Playing)
                {
                    snake.push_front(head);
                    if (head == food)
                    {
                        score++;
                        food = randomFood(snake, cfg); // eat: spawn new food, don't pop tail
                    }
                    else
                    {
                        snake.pop_back(); // normal move: remove tail
                    }
                }
            }
        }

        // --- Draw ---
        window.clear(sf::Color(20, 20, 20));

        // ===== MENU =====
        if (state == GameState::Menu)
        {
            window.draw(titleText);
            window.draw(labelSpeed);
            window.draw(labelGrid);
            window.draw(labelColor);
            window.draw(labelWalls);

            // Update selected/hover state then draw each group
            auto drawGroup = [&](std::vector<Button> &btns, int sel)
            {
                for (int i = 0; i < (int)btns.size(); i++)
                {
                    btns[i].setSelected(i == sel);
                    btns[i].setHovered(btns[i].contains(mousePos) && i != sel);
                    btns[i].draw(window);
                }
            };
            drawGroup(speedBtns, speedSel);
            drawGroup(gridBtns, gridSel);
            drawGroup(colorBtns, colorSel);
            drawGroup(wallBtns, wallSel);

            playBtn.setSelected(false);
            playBtn.setHovered(playBtn.contains(mousePos));
            playBtn.draw(window);
        }

        // ===== PLAYING + GAME OVER =====
        if (state == GameState::Playing || state == GameState::GameOver)
        {
            // Draw faint vertical lines
            for (int x = 0; x < WIDTH; x += cfg.cell)
            {
                sf::RectangleShape line(sf::Vector2f(1, HEIGHT));
                line.setPosition(sf::Vector2f(x, 0));
                line.setFillColor(sf::Color(40, 40, 40));
                window.draw(line);
            }

            // Draw faint horizontal lines
            for (int y = 0; y < HEIGHT; y += cfg.cell)
            {
                sf::RectangleShape line(sf::Vector2f(WIDTH, 1));
                line.setPosition(sf::Vector2f(0, y));
                line.setFillColor(sf::Color(40, 40, 40));
                window.draw(line);
            }

            // Snake - lerp each channel from head color to tail color
            for (size_t i = 0; i < snake.size(); i++)
            {
                float t = 1.f - (float)i / snake.size(); // 1.0 at head, ~0.0 at tail

                cell.setFillColor(sf::Color(
                    static_cast<uint8_t>(cfg.r * t + cfg.tr * (1 - t)), // red channel
                    static_cast<uint8_t>(cfg.g * t + cfg.tg * (1 - t)), // green channel
                    static_cast<uint8_t>(cfg.b * t + cfg.tb * (1 - t))  // blue channel
                ));
                cell.setPosition(sf::Vector2f(snake[i].x * cfg.cell + 1.f, snake[i].y * cfg.cell + 1.f));
                window.draw(cell);
            }

            // Food — unchanged
            cell.setFillColor(sf::Color(220, 50, 50));
            cell.setPosition(sf::Vector2f(food.x * cfg.cell + 1.f, food.y * cfg.cell + 1.f));
            window.draw(cell);

            // Score — unchanged
            scoreText.setString("Score: " + std::to_string(score));
            window.draw(scoreText);

            // Game over overlay — now also shows "R = menu"
            if (state == GameState::GameOver)
            {
                // Semi-transparent dark overlay
                sf::RectangleShape overlay(sf::Vector2f(WIDTH, HEIGHT));
                overlay.setFillColor(sf::Color(0, 0, 0, 150));
                window.draw(overlay);

                // Centered message
                msgText.setString("GAME OVER  |  R for menu");
                sf::FloatRect bounds = msgText.getLocalBounds();
                msgText.setPosition({WIDTH / 2.f - bounds.size.x / 2.f,
                                     HEIGHT / 2.f - bounds.size.y / 2.f});
                window.draw(msgText);
            }
        }

        window.display();
    }
}