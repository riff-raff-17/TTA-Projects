# Session 3 Teaching Notes: Randomness, User Input & Validation with Turtle

**Builds on:** `2_turtle_functions_and_loops.py`, `2a_flower_project.py` (Session 2 — variables, functions, loops, lists, conditionals)
**Files for this session:** `3_turtle_random_and_input.py`, `3a_garden_project.py`
**Length:** written for ~2h15–2h30. This session picked up several extra turtle features (fill, circles/arcs, dots, text) on top of its core topics, so it runs a little longer than Session 2. Sections marked **OPTIONAL** in the agenda are the first to trim or move to take-home if you're short on time — see the note at the end of each optional section for what's safe to cut.

---

## Learning Objectives

By the end of this session, students should be able to:

1. Use `random.randint()`, `random.choice()`, and `random.random()` to introduce controlled randomness into a program.
2. Fill a shape with color using `begin_fill()` / `end_fill()`, and set pen color and fill color independently.
3. Draw circles and arcs with `circle()`, and place a marker with `dot()`.
4. Add text to a drawing with `write()`.
5. Explain the difference between `input()` (terminal) and `screen.numinput()` / `screen.textinput()` (turtle GUI dialogs).
6. Validate user input with `if`/`else` (and a `while` loop) so the program doesn't crash or misbehave on bad input.
7. Combine a loop with random choices to build a "random walk."
8. Combine all of the above into one project.

---

## Agenda

| Time | Section | Flex | Activity |
| --- | --- | --- | --- |
| 0:00–0:08 | 1. Recap | Core | Re-run `2a_flower_project.py`, quick Q&A on Session 2 |
| 0:08–0:20 | 2. The `random` module + dice tally | Core | `randint`, `choice`, `random()` |
| 0:20–0:35 | 3. Random colors + filling shapes | Core | `begin_fill()` / `end_fill()` in `draw_flower` |
| 0:35–0:43 | 4. Pen color vs. fill color | **Optional** | `pencolor()` / `fillcolor()` independently |
| 0:43–0:53 | 5. Randomizing multiple parameters | **Optional** | A row of flowers with random sides/size |
| 0:53–1:08 | 6. Circles, arcs & dots | Core | `circle()`, `circle(radius, extent)`, `dot()` |
| 1:08–1:23 | 7. User input | Core | `input()` vs `numinput()` / `textinput()` |
| 1:23–1:38 | 8. Validation | Core | `if`/`else` fallback, `while` loop, validated text input |
| 1:38–1:43 | **Break** | | |
| 1:43–1:53 | 9. Labeling with `write()` | **Optional** | On-canvas text |
| 1:53–2:08 | 10. Random walk | Core | Loop with random direction/distance, color-changing bonus |
| 2:08–2:28 | 11. Mini project | Core | `garden_project.py` |
| 2:28–2:35 | 12. Wrap-up | Core | Recap, hand out challenges |

**If you need to land closer to 2 hours:** cut sections 4, 5, and 9 (they're demonstrated in the script but nothing later in the session depends on them), and treat the random-walk color-change bonus in section 10 as a "here's a quick extra" rather than something to build live.

---

## Section-by-Section Notes

### 1. Recap (8 min)

Re-run last session's flower project. Ask students to point at the code and name the concept: "which part is the loop? the function? the list? the conditional?" This session touches every one of those, plus several new turtle tools, so a quick refresher earns its keep.

### 2. The `random` module (12 min)

Introduce three functions with simple `print()` demos before touching turtle at all:

- `random.randint(1, 6)` — a random whole number in a range (dice-roll analogy works well).
- `random.choice([...])` — a random pick from a list. Connect it explicitly to `colors[i % len(colors)]` from last session: same idea (pick from a list), different mechanism (random instead of cycling in order).
- `random.random()` — a random float between 0 and 1.

**Dice tally (new):** roll a die 20 times, tally results in a list, print a simple bar chart with `*` characters. Nothing syntactically new here — it's a checkpoint that random, loops, and lists all still work together. Ask: does every face come up the same number of times? Run it again — same pattern? Good moment to say "random doesn't mean evenly spread out over a small sample."

**Common pitfall:** forgetting `import random` at the top, or writing `random.randint` as `randint.random`.

### 3. Random colors + filling shapes (15 min)

First replace `colors[i % len(colors)]` in `draw_flower` with `random.choice(colors)` — same swap as before. Then introduce `begin_fill()` / `end_fill()`: everything drawn between the two calls gets filled with the current fill color once the shape closes.

**Common pitfall:** forgetting `end_fill()` — the outline still draws but nothing fills in, which reads as "fill didn't work" rather than "I forgot a line." Also, `begin_fill()` must come *before* the shape is drawn, not after.

**Talking point:** this is a strong visual payoff for very little new syntax — worth pausing on the before/after (outline flower vs. filled flower) so students feel the win.

### 4. OPTIONAL — Pen color vs. fill color (8 min)

`color()` sets both pen and fill to the same value. `pencolor()` and `fillcolor()` let you set them independently — e.g., a black outline around a gold fill. Also mention `color(pencolor, fillcolor)` as a one-line shortcut for the same thing.

**If cutting for time:** skip this section entirely — `draw_flower` already works fine using `color()` for both, and nothing later depends on separating them.

### 5. OPTIONAL — Randomizing multiple parameters at once (10 min)

Extend the "one random color" idea to randomizing `sides` and `size` too, drawing a small row of flowers so students see several independent combinations side by side.

**Talking point:** each call to `random.randint()` is independent — `sides` and `size` don't have to "match" each other in any way. This previews exactly what the mini project does with position added as a third random parameter.

**If cutting for time:** this is nice-to-have but not load-bearing — the mini project reintroduces the same idea from scratch.

### 6. Circles, arcs, and dots (15 min)

- `circle(radius)` draws a full circle — no loop needed. Contrast this with `draw_polygon`, which needed a loop to approximate a shape; a circle is "free."
- `circle(radius, extent)` draws just part of one — `extent` is the arc's angle in degrees (`circle(40, 90)` is a quarter-circle).
- Introduce `draw_circle_flower` as an alternative to `draw_flower`: same loop-and-rotate pattern, but circles instead of polygons, so `draw_polygon` isn't needed at all for this version.
- `dot(diameter, color)` leaves a solid filled circle at the turtle's current position — used here to mark a flower's center.

**Common pitfall:** mixing up which `circle()` argument is the radius vs. the extent; also, `circle()` uses **radius**, while polygon `size` in `draw_polygon` is a **side length** — worth naming that distinction explicitly since students will otherwise expect them to behave the same way.

**Good check for understanding:** ask a student to predict what `draw_circle_flower` will look like with `petals=4` vs. `petals=20` before running it.

### 7. User Input (15 min)

Two ways to ask the user for something:

- `input("How many petals? ")` — plain Python, appears in the terminal, always returns a **string** (a common gotcha: `int(input(...))` is needed for math).
- `screen.numinput(title, prompt, default, minval, maxval)` — a turtle GUI popup, returns a number directly (or `None` if cancelled).
- `screen.textinput(title, prompt)` — same idea, for free text (e.g., a color name).

**Talking point:** GUI input feels friendlier for a turtle program, but terminal `input()` is worth showing too since it's what they'll meet in almost every other Python context.

### 8. Validation (15 min)

- Simple version: `if petals is None or petals < 3: petals = 12` (fall back to a default).
- Stronger version: a `while` loop that keeps asking until the answer is valid — left **active** in the script (not commented out) so students see the dialog reappear immediately on a bad answer.
- Validating *text* input is a different shape of problem than validating a *number*: instead of a min/max, it's checking membership in an allowed list (`.strip().lower()` first, then `in allowed_colors`).

**Common pitfall:** students compare `petals == None` instead of `petals is None`; also forgetting `.lower()` on text input, so `"Red"` fails a check against `"red"` even though a person would call them the same thing.

**Good check for understanding:** ask "what happens right now if I click Cancel on the popup?" before adding the `while` loop, then again after — the fix should be visibly different.

### Break (5 min)

### 9. OPTIONAL — Labeling with `write()` (10 min)

`write(text, font=(...))` drops text at the turtle's current position — typically `penup()`, `goto()` where the label should go, then `write()`. Demonstrate labeling a flower with its petal count and size.

**If cutting for time:** skip and fold it directly into the mini project instead, where it's used once for a garden title.

### 10. Random Walk (15 min)

Same "loop that changes a value each time" idea from Session 2, now with a random value instead of a fixed pattern:

```python
for _ in range(50):
    angle = random.randint(0, 360)
    distance = random.randint(10, 40)
    walker.setheading(angle)
    walker.forward(distance)
```

Ask students to predict the shape of the path before running it, then compare a few runs side by side.

**Bonus (time-permitting):** a second random walk that also changes pen color and thickness every 10 steps, using the `i % 10 == 0` conditional pattern from Session 2 — a nice reminder that a few familiar pieces (loop, `if`, function call) combine into something that looks a lot more complex than any one piece alone.

### 11. Mini Project: Customizable Random Garden (20 min)

Open `garden_project.py`. This combines everything from the session:

- Validated input for how many flowers to plant.
- Each flower gets a random position, petal count, and size, and randomly uses either `draw_flower` (polygon petals) or `draw_circle_flower` (circle petals).
- A `dot()` marks every flower's center.
- A single `write()` call labels the whole garden with the flower count.

Run it once as-is, then invite students to change the ranges or add their own petal style. The commented-out **Challenge extensions** at the bottom are for early finishers or homework.

### 12. Wrap-up (7 min)

Recap the objectives in one sentence each. Send students off with the challenge extensions, plus the take-home challenges below if you want more.

---

## Take-Home / Challenge Exercises

- **Weighted colors:** look up `random.choices(colors, weights=[...])` and make some colors more likely to appear than others.
- **Bounded random walk:** modify the random walk so the turtle turns back toward the center whenever it strays past a certain distance from `(0, 0)`.
- **Fully validated garden:** extend `garden_project.py` so *every* numeric input (flower count, size range) is validated with a `while` loop, not just petal count.
- **Random walk race:** create two turtles with different colors, run a random walk for each, and see which one travels farther from the start (`turtle.distance(0, 0)`).
- **Circle-only garden:** modify `garden_project.py` so every flower uses `draw_circle_flower` instead of randomly picking between the two styles, then experiment with overlapping radii for a denser look.
- **Per-flower labels:** uncomment the labeling extension in `garden_project.py` so every flower shows its own petal count instead of one garden-wide title.

---

## Held Off for Session 4 (and Beyond)

These turtle features came up when discussing "what else can turtle do," but fit Session 4's interactivity theme better than this one — introducing them now would either front-run their real use case or add complexity without a payoff yet:

- **`screen.onkey()` / `screen.onclick()` / `screen.ontimer()`** — the core of Session 4's "make it responsive" theme. Nothing in Session 3 needed the turtle to react to anything.
- **`shape()` / `shapesize()`** — changes what the turtle icon looks like. Most useful once the turtle is visible and moving *purposefully* (e.g., a player or game piece), not while it's hidden as a drawing pen.
- **`screen.tracer(0)` + `screen.update()`** — manual control over when the screen redraws. Pairs naturally with animation/game loops in Session 4; introducing it now (with static drawings) wouldn't have an obvious payoff.
- **`position()`, `heading()`, `distance()`, `towards()`** — turtle state queries. These are exactly what's needed for "chase" or "bounce off the wall" logic, which is Session 4 territory. (`distance()` is mentioned in this session's take-home list, but only as something to look up, not taught directly.)
- **`clone()`** — spawns a copy of a turtle. Most useful once there's a reason for several turtles to act independently, e.g., multiple game entities in Session 4.
- **`screen.bgpic()`** — background images. A nice-to-have polish item with no dependency on anything else; low priority, can be mentioned in passing whenever it's convenient.
- **Intro to classes** and **recursion** — still further out, as noted in Session 2's notes; best introduced once functions and reuse feel fully automatic.
