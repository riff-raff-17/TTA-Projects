"""
Mini Project: Customizable Random Garden
------------------------------------------
Combines everything from this session -- the random module, user
input, validation, and the extra turtle tools (fill, circle
petals, dot centers, write() labels) -- with draw_polygon and
draw_flower from Session 2's flower_project.py.
"""

import random
import turtle

screen = turtle.Screen()
screen.title("Random Garden")
screen.bgcolor("skyblue")

t = turtle.Turtle()
t.speed(0)
t.hideturtle()


# --- Shape-drawing functions ---

def draw_polygon(turtle_obj, sides, size):
    """Draw a regular polygon with the given number of sides and side length."""
    angle = 360 / sides
    for _ in range(sides):
        turtle_obj.forward(size)
        turtle_obj.right(angle)


def draw_flower(turtle_obj, petals, sides, size, colors):
    """Polygon-petaled flower: filled, with a random color per petal."""
    turn_between_petals = 360 / petals
    for _ in range(petals):
        turtle_obj.color(random.choice(colors))
        turtle_obj.begin_fill()
        draw_polygon(turtle_obj, sides, size)
        turtle_obj.end_fill()
        turtle_obj.right(turn_between_petals)


def draw_circle_flower(turtle_obj, petals, radius, colors):
    """Circle-petaled flower: an alternative look, filled, random color per petal."""
    turn_between_petals = 360 / petals
    for _ in range(petals):
        turtle_obj.fillcolor(random.choice(colors))
        turtle_obj.begin_fill()
        turtle_obj.circle(radius)
        turtle_obj.end_fill()
        turtle_obj.right(turn_between_petals)


# --- Validated-input helper ---

def get_valid_number(title, prompt, default, minval, maxval):
    """
    Ask the user for a number; fall back to a default if they
    cancel instead of leaving the program with an invalid value.
    """
    value = screen.numinput(title, prompt, default, minval=minval, maxval=maxval)
    if value is None:
        print(f"No valid answer given for '{prompt}' -- using default {default}.")
        value = default
    return value


# --- Ask the user how many flowers to plant ---

num_flowers = int(get_valid_number(
    "Garden Size", "How many flowers should the garden have? (1-15)", 6, 1, 15
))

flower_colors = ["red", "orange", "gold", "hotpink", "purple", "mediumblue", "white"]
center_colors = ["black", "brown", "saddlebrown"]

for _ in range(num_flowers):
    # Random position within a comfortable window inside the screen
    x = random.randint(-280, 260)
    y = random.randint(-230, 220)

    # Random shape and size for variety between flowers
    petals = random.randint(6, 14)
    size = random.randint(18, 40)

    t.penup()
    t.goto(x, y)
    t.setheading(0)
    t.pendown()
    t.pensize(random.randint(1, 3))

    # Randomly pick which petal style this flower gets: polygon or circle
    if random.choice([True, False]):
        sides = random.choice([3, 4, 5, 6])
        draw_flower(t, petals=petals, sides=sides, size=size, colors=flower_colors)
    else:
        draw_circle_flower(t, petals=petals, radius=size // 2, colors=flower_colors)

    # A dot marks the center of every flower, regardless of petal style
    t.penup()
    t.goto(x, y)
    t.pendown()
    t.dot(10, random.choice(center_colors))


# --- Label the garden ---

label = turtle.Turtle()
label.hideturtle()
label.penup()
label.goto(-290, 280)
label.write(
    f"A garden of {num_flowers} random flowers",
    font=("Arial", 16, "bold"),
)


# ---------------------------------------------------------------
# Challenge extensions -- try uncommenting one at a time
# ---------------------------------------------------------------

# 1. Fully validate every input, not just flower count:
# min_size = int(get_valid_number("Min Size", "Smallest petal size? (10-40)", 20, 10, 40))
# max_size = int(get_valid_number("Max Size", "Largest petal size? (41-80)", 50, 41, 80))
# size = random.randint(min_size, max_size)  # use this instead of the fixed range above

# 2. Weighted colors (some colors more common than others):
# weighted_colors = random.choices(
#     flower_colors,
#     weights=[3, 3, 1, 2, 1, 2, 1],  # must match length of flower_colors
#     k=petals,
# )

# 3. Label each flower with its own petal count instead of one
#    garden-wide title:
# t.penup()
# t.goto(x, y - size - 15)
# t.pendown()
# t.write(str(petals), font=("Arial", 8, "normal"))

# 4. Make every flower a circle-flower and experiment with
#    overlapping radii for a denser, more layered look:
# draw_circle_flower(t, petals=petals, radius=size, colors=flower_colors)

# 5. Avoid overlapping flowers by keeping a list of used positions
#    and re-rolling x, y if a new flower is too close to an old one:
# used_positions = []
# def far_enough(x, y, positions, min_distance=60):
#     return all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 > min_distance for px, py in positions)


turtle.done()
