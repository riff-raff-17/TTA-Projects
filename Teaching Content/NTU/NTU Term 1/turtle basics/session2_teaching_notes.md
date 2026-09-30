# Session 2 Teaching Notes: Functions, Loops, Lists & Conditionals with Turtle

**Builds on:** `turtle_basics.py` (Session 1 — screen setup, movement, pen up/down, color, a basic loop)
**Files for this session:** `turtle_functions_and_loops.py`, `flower_project.py`
**Length:** ~2 hours, including one break

---

## Learning Objectives

By the end of this session, students should be able to:

1. Replace hardcoded numbers with variables and explain why that's useful.
2. Define a function with parameters, and call it with different arguments.
3. Use a `for` loop to call a function repeatedly instead of copy-pasting code.
4. Create a list, and loop over it with `for x in my_list`.
5. Use `if` / `else` to make a program behave differently based on a condition.
6. Combine all of the above into one small project.

---

## Agenda

| Time | Section | Activity |
| --- | --- | --- |
| 0:00–0:05 | Recap | Quick Q&A on Session 1, re-run `turtle_basics.py` |
| 0:05–0:15 | Variables | Refactor the square loop to use variables |
| 0:15–0:40 | Functions & parameters | Build `draw_square(turtle_obj, size)` live |
| 0:40–0:55 | Nested loops | Draw a growing row of squares using the function |
| 0:55–1:00 | **Break** | |
| 1:00–1:15 | Lists | Introduce a color list, loop over it |
| 1:15–1:30 | Conditionals | `if` / `else` based on even/odd loop index |
| 1:30–2:00 | Mini project | `flower_project.py` — build and customize together |
| 2:00–2:10 | Wrap-up | Recap, hand out challenge exercises |

Treat the timings as a guide, not a contract — if functions take longer to click, let them; the mini-project section can flex shorter.

---

## Section-by-Section Notes

### 1. Recap (5 min)

Ask students to explain, in their own words: what does `forward()` do? What's the difference between `penup()` and `pendown()`? What did the loop in `turtle_basics.py` do? This surfaces anyone who's shaky before you build on top of it.

### 2. Variables (10 min)

Take the square-drawing loop from last time and pull the `100` and `90` out into named variables (`side_length`, `turn_angle`). Change the values and re-run.

**Key point:** a variable is a name for a value. It doesn't change *what* the code does, just makes it easier to read and to change in one place instead of four.

**Common question:** "Isn't this just extra typing?" — Good moment to say: it pays off once the same number appears in multiple places, or once you want to change it based on user input (which is coming later).

### 3. Functions & Parameters (25 min)

This is the conceptual centerpiece of the session — spend real time here.

- Start with the *problem*: we want several squares of different sizes. Copy-pasting the loop each time is tedious and error-prone.
- Introduce `def draw_square(turtle_obj, size):` and walk through:
  - `def` starts a function definition.
  - Everything indented underneath is the function's body.
  - `turtle_obj` and `size` are **parameters** — they don't have values yet, they're placeholders.
  - When you *call* `draw_square(t, 40)`, `40` becomes the value of `size` for that run.
- Live-code drawing two squares of different sizes with two calls to the same function.

**Common pitfall:** students forget to pass `t` (the turtle object) and wonder why nothing draws, or they mix up definition order (calling before defining). Also expect confusion between the parameter name (`size`) and the argument value (`40`) — worth explicitly naming that distinction once.

**Good check for understanding:** ask a student to add a `draw_triangle` function themselves, using `draw_square` as a template.

### 4. Nested Loops (15 min)

Now use a `for` loop to *call* the function multiple times, changing `size` each iteration:

```python
for i in range(5):
    size = 20 + i * 15
    draw_square(t, size)
```

This is where "functions + loops" clicks for a lot of students — the loop replaces what would have been five separate function calls.

**Talking point:** contrast "a loop that repeats the same fixed code" (Session 1) with "a loop that calls a function with a *different* value each time" (this session). That's the upgrade.

### 5. Break (5 min)

### 6. Lists (15 min)

Introduce a list as a container: `colors = ["red", "blue", "green", "purple", "orange"]`.

- Show `for color in colors:` — the loop variable takes each value in turn.
- Combine with the function from before: a `draw_colored_square` function that sets the color, then draws.

**Common question:** "What's the difference between this and the loop from before?" — Before we looped over *numbers* (`range(5)`), now we're looping directly over the *values we care about* (colors). Both are valid; which one to use depends on what you're iterating over.

### 7. Conditionals (15 min)

Introduce `if` / `else` using the loop's index and the modulo operator (`i % 2 == 0`) to alternate colors. If `%` hasn't come up before, take two minutes to explain it as "remainder after division" with a couple of numeric examples (`5 % 2 == 1`, `4 % 2 == 0`) before applying it to turtle.

**Common pitfall:** forgetting the colon after `if condition:`, or indentation mismatches between the `if` and `else` blocks — a good moment to reinforce that Python uses indentation to mark blocks, the same as function bodies and loop bodies.

### 8. Mini Project: Polygon Flower (30 min)

Open `flower_project.py`. This ties every concept from the session together:

- `draw_polygon(turtle_obj, sides, size)` — a generalized version of `draw_square`, using `360 / sides` for the turn angle. Ask: "what does this become when `sides = 4`?" to connect it back to the square function.
- `draw_flower(turtle_obj, petals, sides, size, colors)` — a loop that draws multiple polygons rotated around a point, picking a color from the list each time with `colors[i % len(colors)]`.

Run it once as-is, then invite students to change `petals`, `sides`, `size`, or the color list and re-run. The three commented-out **Challenge extensions** at the bottom (random colors, a second flower, asking for input) are there for early finishers or as homework — uncomment one at a time so failures are easy to isolate.

### 9. Wrap-up (10 min)

Recap the five concepts in one sentence each. Send students off with the challenge extensions, plus the take-home challenges below if you want more.

---

## Take-Home / Challenge Exercises

- **Spiral:** write a loop that draws a square, turns slightly (e.g. 4 degrees), and grows the size each time — no functions required beyond what's already written.
- **Random flower:** uncomment the random-color extension in `flower_project.py` and make every petal a random color.
- **User-controlled polygon:** use `screen.numinput(...)` to ask the user how many sides to draw, then call `draw_polygon` with that value.
- **Two functions, one drawing:** write a `draw_star` function (using `t.right(144)` after each `forward` traces a five-pointed star) and call it alongside `draw_square` in the same picture.

---

## Ideas for Future Sessions

- **Randomness:** a deeper dive into the `random` module — `random.randint`, `random.choice`, `random.random()` — building a "random walk" turtle.
- **User input & validation:** `input()` and `screen.numinput()`/`screen.textinput()`, plus `if` statements that check the input is valid before using it.
- **Recursion:** recursive tree or fractal drawing with turtle — a natural, visual introduction to recursion.
- **Simple interactivity:** `screen.onkey()` / `screen.onclick()` to make the turtle respond to keyboard or mouse input (stepping stone toward a very simple game).
- **Intro to classes:** wrap turtle behavior in a small custom class (e.g. a `Robot` class with a `.draw_shape()` method) as a first taste of object-oriented programming, once functions feel solid.
