# Conway's Game of Life — Code Walkthrough

A breakdown of `game_of_life.cpp`, a Conway's Game of Life implementation in C++ using SFML.

---

## Overview

The program simulates Conway's Game of Life on a 2D grid. Cells are either alive or dead, and each generation the grid evolves according to four simple rules:

- A live cell with fewer than 2 live neighbors **dies** (underpopulation).
- A live cell with 2 or 3 live neighbors **survives**.
- A live cell with more than 3 live neighbors **dies** (overpopulation).
- A dead cell with exactly 3 live neighbors **becomes alive** (reproduction).

Despite their simplicity, these rules produce remarkably complex and unpredictable behavior.

---

## Data Structures

```cpp
using Grid = std::vector<std::vector<bool>>;
```

The grid is a 2D vector of booleans — `true` means alive, `false` means dead. It's indexed as `grid[x][y]`, where `x` is the column and `y` is the row.

```cpp
Grid makeGrid(int cols, int rows)
{
    return Grid(cols, std::vector<bool>(rows, false));
}
```

`makeGrid` creates a blank grid of the given dimensions, with every cell initialized to dead. It's called at startup and whenever the user changes the zoom level or presses C to clear.

---

## Grid Helpers

### `randomize`

```cpp
void randomize(Grid& g, int cols, int rows)
{
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
            g[x][y] = (std::rand() % 4 == 0); // ~25% alive
}
```

Iterates every cell and sets it alive with a 25% probability. This density tends to produce interesting initial patterns — low enough to avoid immediate stagnation, high enough for complex interactions.

### `countNeighbors`

```cpp
int countNeighbors(const Grid& g, int x, int y, int cols, int rows)
{
    int count = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + cols) % cols; // wrap edges
            int ny = (y + dy + rows) % rows;
            if (g[nx][ny]) count++;
        }
    return count;
}
```

Counts live neighbors by checking all 8 surrounding cells (the 3×3 neighborhood minus the center cell itself). The key detail is the **toroidal wrapping**: instead of treating the edges as hard boundaries, cells on the left edge are neighbors with cells on the right edge, and similarly for top and bottom. This is done with modular arithmetic — `(x + dx + cols) % cols` — which ensures the index never goes out of bounds and the grid behaves like the surface of a donut.

### `step`

```cpp
Grid step(const Grid& current, int cols, int rows)
{
    Grid next = makeGrid(cols, rows);
    for (int x = 0; x < cols; x++)
        for (int y = 0; y < rows; y++)
        {
            int n = countNeighbors(current, x, y, cols, rows);
            if (current[x][y])
                next[x][y] = (n == 2 || n == 3); // survive
            else
                next[x][y] = (n == 3);             // born
        }
    return next;
}
```

This is the heart of the simulation. It takes the current grid and produces the next generation.

The critical technique here is the **double-buffer pattern**: `step` always reads from `current` and writes to a brand new grid `next`. It never modifies `current` while reading it. Without this, you'd compute some cells' next states based on already-updated neighbors — which would be incorrect and produce a different (wrong) simulation.

The four Game of Life rules are condensed into two lines:
- `(n == 2 || n == 3)` — a live cell survives only with 2 or 3 neighbors.
- `(n == 3)` — a dead cell is born only with exactly 3 neighbors.

---

## Main Loop

### Setup

```cpp
int cellSize = 10;
int cols = WIDTH  / cellSize;
int rows = HEIGHT / cellSize;

Grid grid = makeGrid(cols, rows);
randomize(grid, cols, rows);
```

The grid dimensions are derived from the window size and the cell size in pixels. With an 800×600 window and `cellSize = 10`, you get an 80×60 grid — 4,800 cells.

`running` starts as `false` so the simulation is paused on launch, letting you inspect the random state or draw your own patterns before starting.

### Event Handling

Three categories of input are handled:

**Keyboard:**

| Key     | Action                                    |
|---------|-------------------------------------------|
| `Space` | Toggle play / pause                       |
| `R`     | Randomize the grid, reset generation count |
| `C`     | Clear the grid (all dead), reset count    |

**Scroll wheel** — zooms in and out by changing `cellSize`, then rebuilds the grid:

```cpp
cellSize = std::clamp(cellSize + (int)w->delta, 4, 40);
cols = WIDTH  / cellSize;
rows = HEIGHT / cellSize;
grid = makeGrid(cols, rows);
randomize(grid, cols, rows);
```

`std::clamp` keeps `cellSize` in the range [4, 40] so cells stay visible but not absurdly large.

**Mouse painting** — lets you draw on the grid while paused:

```cpp
// On press: decide whether to paint alive or dead
paintVal = !grid[cx][cy]; // toggle: if alive paint dead, vice versa

// Each frame while held:
if (painting)
    grid[cx][cy] = paintVal;
```

The `paintVal` is decided once on the initial click — if you clicked a live cell, dragging will erase; if you clicked a dead cell, dragging will draw. This makes it easy to paint structures like gliders by hand.

### Simulation Tick

```cpp
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
```

The simulation runs at a fixed rate of one generation per `simSpeed` seconds (0.1s by default, so ~10 generations per second), regardless of the display framerate. This is the same time-accumulator pattern used in your Snake game. `dt` is the real elapsed time since the last frame, accumulated until it exceeds the target interval.

### Drawing

```cpp
cellShape.setSize({(float)cellSize - 1.f, (float)cellSize - 1.f});

for (int x = 0; x < cols; x++)
    for (int y = 0; y < rows; y++)
        if (grid[x][y])
        {
            alive++;
            cellShape.setPosition({(float)(x * cellSize), (float)(y * cellSize)});
            cellShape.setFillColor(sf::Color(100, 220, 130));
            window.draw(cellShape);
        }
```

Only live cells are drawn — dead cells are just the dark background. The `-1.f` on the cell size creates a 1-pixel gap between cells, giving the grid a visible structure without drawing explicit grid lines.

The HUD at the top-left is updated each frame with the current state, generation count, and live cell count (which is conveniently computed during the draw loop above).

---

## Design Decisions

**Why `vector<vector<bool>>` instead of a flat array?**
It keeps the indexing intuitive (`grid[x][y]`) and closely matches the mathematical model. A flat `bool grid[cols * rows]` with manual index math (`grid[y * cols + x]`) would be marginally faster but much harder to read. For a grid this size, the performance difference is irrelevant.

**Why does `step` return a new `Grid` instead of modifying in place?**
As explained above, correctness requires reading the old state and writing the new state in separate passes. Returning by value is clean and lets the compiler optimize the move rather than copying. Alternatively you could keep two grids and swap them each generation — that's slightly more efficient but more code.

**Why does scroll zoom reset and re-randomize the grid?**
Resizing the grid while preserving the existing pattern would require resampling or cropping, which adds meaningful complexity. For a two-hour project, a clean reset is the right tradeoff.

---

## Possible Extensions

- **Variable simulation speed** — add `[` and `]` keys to halve or double `simSpeed`.
- **Step mode** — press `N` while paused to advance exactly one generation at a time.
- **Save / load patterns** — write the grid to a `.cells` or `.rle` file (standard Game of Life formats).
- **Color by age** — track how many generations each cell has been alive and tint accordingly, producing a heatmap effect.
- **Multiple food colors** — currently all live cells are the same green; you could color them based on neighbor count to visualize the rules directly.