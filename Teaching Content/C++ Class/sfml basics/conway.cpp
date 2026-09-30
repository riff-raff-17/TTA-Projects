// Conway's Game of Life - SFML
// Controls: Space = play/pause, R = random fill, C = clear,
// Click/drag = toggle cells, Scroll = zoom (cell size)

#include <SFML/Graphics.hpp>
#include <vector>
#include <cstdlib>
#include <string>

const int WIDTH = 800, HEIGHT = 600;

// --- Grid helpers ---
using Grid = std::vector<std::vector<bool>>;

Grid makeGrid(int cols, int rows)
{
    return Grid(cols, std::vector<bool>(rows, false));
}

void randomize(Grid &g, int cols, int rows)
{
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
            g[x][y] = (std::rand() % 4 == 0); // ~25% alive
}

int countNeighbors(const Grid &g, int x, int y, int cols, int rows)
{
    int count = 0;
    for (int dx = -1; dx <= 1; dx ++) // outer loop
    {
        for (int dy = -1; dy <= 1; dy++) // inner loop
        {
            if (dx == 0 && dy == 0)
                continue;
            int nx = (x + dx + cols) % cols; // wrap edges
            int ny = (y + dy + rows) % rows;
            if (g[nx][ny])
                count++;
        }
    }
    return count;
}

Grid step(const Grid &current, int cols, int rows)
{
    Grid next = makeGrid(cols, rows);
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
        {
            int n = countNeighbors(current, x, y, cols, rows);
            if (current[x][y])
                next[x][y] = (n == 2 || n == 3); // survive
            else
                next[x][y] = (n == 3); // born
        }
    return next;
}

// --- Main ---
int main()
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    sf::RenderWindow window(sf::VideoMode({WIDTH, HEIGHT}), "Game of Life");
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.openFromFile("FiraCodeNerdFont-Regular.ttf"))
        return -1;

    int cellSize = 10;
    int cols = WIDTH / cellSize;
    int rows = HEIGHT / cellSize;

    Grid grid = makeGrid(cols, rows);
    randomize(grid, cols, rows);

    bool running = false;  // paused on start so you can draw first
    bool painting = false; // true while mouse button held
    bool paintVal = true;  // alive or dead while painting

    float simTimer = 0.f;
    float simSpeed = 0.1f; // seconds per generation
    int generation = 0;

    sf::Clock clock;
    sf::RectangleShape cellShape;

    sf::Text hud(font);
    hud.setCharacterSize(16);
    hud.setFillColor(sf::Color::White);
    hud.setPosition({6.f, 4.f});

    while (window.isOpen())
    {
        float dt = clock.restart().asSeconds();
        sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

        // --- Events ---
        while (const auto event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();

            // Keyboard
            if (const auto *k = event->getIf<sf::Event::KeyPressed>())
            {
                if (k->code == sf::Keyboard::Key::Space)
                    running = !running;
                if (k->code == sf::Keyboard::Key::R)
                {
                    randomize(grid, cols, rows);
                    generation = 0;
                }
                if (k->code == sf::Keyboard::Key::C)
                {
                    grid = makeGrid(cols, rows);
                    generation = 0;
                }
            }

            // Scroll wheel - resize cells and rebuild grid
            if (const auto *w = event->getIf<sf::Event::MouseWheelScrolled>())
            {
                cellSize = std::clamp(cellSize + (int)w->delta, 4, 40);
                cols = WIDTH / cellSize;
                rows = HEIGHT / cellSize;
                grid = makeGrid(cols, rows);
                randomize(grid, cols, rows);
                generation = 0;
            }

            // Mouse press - start painting
            if (const auto *mb = event->getIf<sf::Event::MouseButtonPressed>())
            {
                if (mb->button == sf::Mouse::Button::Left)
                {
                    painting = true;
                    int cx = (int)mousePos.x / cellSize;
                    int cy = (int)mousePos.y / cellSize;
                    if (cx >= 0 && cx < cols && cy >= 0 && cy < rows)
                        paintVal = !grid[cx][cy]; // toggle: if alive paint dead, vice versa
                }
            }
            if (const auto *mb = event->getIf<sf::Event::MouseButtonReleased>())
                if (mb->button == sf::Mouse::Button::Left)
                    painting = false;
        }

        // Drag-paint
        if (painting)
        {
            int cx = (int)mousePos.x / cellSize;
            int cy = (int)mousePos.y / cellSize;
            if (cx >= 0 && cx < cols && cy >= 0 && cy < rows)
                grid[cx][cy] = paintVal;
        }

        // --- Simulate ---
        if (running)
        {
            simTimer += dt;
            if (simTimer >= simSpeed)
            {
                simTimer = 0.f;
                grid = step(grid, cols, rows);
                generation++;
            }
        }

        // --- Draw ---
        window.clear(sf::Color(15, 15, 20));

        cellShape.setSize({(float)cellSize - 1.f, (float)cellSize - 1.f});

        int alive = 0;
        for (int x = 0; x < cols; x++)
            for (int y = 0; y < rows; y++)
                if (grid[x][y])
                {
                    alive++;
                    cellShape.setPosition({(float)(x * cellSize), (float)(y * cellSize)});
                    cellShape.setFillColor(sf::Color(100, 220, 130));
                    window.draw(cellShape);
                }

        // HUD
        hud.setString(
            (running ? "[SPACE] pause" : "[SPACE] play ") +
            std::string("  [R] random  [C] clear  [scroll] zoom\n") +
            "gen: " + std::to_string(generation) +
            "   alive: " + std::to_string(alive)
        );
        window.draw(hud);

        window.display();
    }
}