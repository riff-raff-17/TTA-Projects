"""
Session 3: Randomness, User Input & Validation with Turtle
--------------------------------------------------------------
Builds directly on 2_turtle_functions_and_loops.py and
2a_flower_project.py from last session.

This version also folds in a handful of turtle features beyond
what Sessions 1-2 covered: filling shapes, circles/arcs, dots, and
on-canvas text. None of these are new PROGRAMMING concepts -- they
are just more tools on the turtle itself -- so they slot in without
introducing new syntax to explain.

TEACHING TIP: Don't run this whole file at once on the first pass.
Build it up section by section live, running after each new section
is added, so students see the change each new idea makes. Sections
marked OPTIONAL are safe to skip or shorten if you're short on time
-- see session3_teaching_notes.md for a suggested trim list.
"""

import random
import turtle

# ---------------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------------

screen = turtle.Screen()
screen.title("Randomness, Input, and Validation")
screen.bgcolor("white")

t = turtle.Turtle()
t.speed(0)


# ---------------------------------------------------------------
# 1. The random module: three tools worth knowing
# ---------------------------------------------------------------
# Run these prints FIRST, before touching turtle, so students see
# what each function returns without a drawing to distract them.

print("random.randint(1, 6)  ->", random.randint(1, 6))       # whole number, both ends included
print("random.choice([...])  ->", random.choice(["red", "blue", "green"]))  # one item from a list
print("random.random()       ->", random.random())            # float between 0.0 and 1.0


# 1b. Dice roll tally: random + loops + lists, all together
# --------------------------------------------------------------
tally = [0, 0, 0, 0, 0, 0, 0]  # index 0 unused, so tally[1] counts 1s, etc.
num_rolls = 20

for _ in range(num_rolls):
    roll = random.randint(1, 6)
    tally[roll] += 1

print(f"\nResults of rolling a die {num_rolls} times:")
for face in range(1, 7):
    print(f"  {face}: {'*' * tally[face]}  ({tally[face]} times)")

# Ask students: does every face come up the same number of times?
# Run it again -- does the pattern look the same? Good moment to
# say "random doesn't mean evenly spread out over a small sample."


# ---------------------------------------------------------------
# 2. Reusing Session 2's functions, now with random colors AND fill
# ---------------------------------------------------------------

def draw_polygon(turtle_obj, sides, size):
    """Draw a regular polygon with the given number of sides and side length."""
    angle = 360 / sides
    for _ in range(sides):
        turtle_obj.forward(size)
        turtle_obj.right(angle)


def draw_flower(turtle_obj, petals, sides, size, colors):
    """
    Draw 'petals' copies of a polygon, each rotated evenly around
    a full circle, picking a random color and FILLING each petal.
    """
    turn_between_petals = 360 / petals
    for _ in range(petals):
        turtle_obj.color(random.choice(colors))  # sets pen AND fill color
        turtle_obj.begin_fill()
        draw_polygon(turtle_obj, sides, size)
        turtle_obj.end_fill()
        turtle_obj.right(turn_between_petals)


t.hideturtle()
petal_colors = ["red", "orange", "gold", "purple", "deeppink", "blue"]
draw_flower(t, petals=12, sides=6, size=60, colors=petal_colors)

# New this session: begin_fill() / end_fill(). Everything drawn
# between the two calls gets filled with the current fill color
# once end_fill() closes the shape.
#
# Common pitfall: forgetting end_fill() -- the outline still draws,
# but nothing fills in, which reads as "fill didn't work" rather
# than "I forgot a line." Also, begin_fill() must come BEFORE the
# shape is drawn, not after.


# ---------------------------------------------------------------
# 3. OPTIONAL -- Pen color vs. fill color, independently
# ---------------------------------------------------------------
# color() sets both pen and fill to the same thing. fillcolor()
# and pencolor() let you set them separately -- e.g. a black
# outline around a gold fill.

t.penup()
t.goto(-250, 250)
t.pendown()
t.pencolor("black")
t.fillcolor("gold")
t.begin_fill()
draw_polygon(t, sides=5, size=50)
t.end_fill()

# t.color("black", "gold") does the same thing in one call --
# color() accepts either one argument (both pen and fill) or two
# (pen, fill separately).


# ---------------------------------------------------------------
# 4. OPTIONAL -- Randomizing more than one thing at once
# ---------------------------------------------------------------
# Color isn't the only thing we can randomize. Here we also
# randomize sides and size, and draw a small row so students see
# several different combinations side by side.

t.penup()
t.goto(-150, 250)
t.pendown()

for _ in range(3):
    random_sides = random.randint(3, 8)
    random_size = random.randint(20, 45)
    draw_flower(t, petals=8, sides=random_sides, size=random_size, colors=petal_colors)
    t.penup()
    t.forward(140)
    t.pendown()

# Talking point: each call to random.randint() is independent --
# sides and size don't have to "match" each other in any way. This
# is exactly what the mini project does next, with position added
# as a third random parameter.


# ---------------------------------------------------------------
# 5. Circles, arcs, and dots
# ---------------------------------------------------------------
# circle(radius) draws a full circle -- no loop needed, unlike
# draw_polygon which needs one to approximate a shape.
# circle(radius, extent) draws just PART of a circle -- extent is
# the arc's angle in degrees.

t.penup()
t.goto(0, 250)
t.setheading(0)
t.pendown()
t.pencolor("black")
t.fillcolor("mediumpurple")
t.begin_fill()
t.circle(40)
t.end_fill()

t.penup()
t.goto(100, 210)
t.pendown()
t.circle(40, 90)  # a quarter-circle arc, not a full circle


def draw_circle_flower(turtle_obj, petals, radius, colors):
    """
    An alternative to draw_flower: petals made of overlapping
    circles instead of polygons. No draw_polygon needed at all --
    circle() does the shape-drawing for us.
    """
    turn_between_petals = 360 / petals
    for _ in range(petals):
        turtle_obj.fillcolor(random.choice(colors))
        turtle_obj.begin_fill()
        turtle_obj.circle(radius)
        turtle_obj.end_fill()
        turtle_obj.right(turn_between_petals)


t.penup()
t.goto(200, 230)
t.setheading(0)
t.pendown()
t.pencolor("black")
draw_circle_flower(t, petals=8, radius=25, colors=petal_colors)
t.dot(14, "black")  # dot(diameter, color) marks the center

# Common pitfall: mixing up which circle() argument is the radius
# vs. the extent. Also worth naming explicitly: circle() takes a
# RADIUS, while draw_polygon's "size" is a SIDE LENGTH -- they
# aren't the same kind of number, even though both control how big
# the shape looks.
#
# Ask students: draw_flower uses draw_polygon; draw_circle_flower
# doesn't need it at all. Same overall pattern (loop + rotate +
# random color), different shape-drawing tool underneath.


# ---------------------------------------------------------------
# 6. User input: terminal vs. turtle GUI dialogs
# ---------------------------------------------------------------
# input() is plain Python and always returns a STRING.
# screen.numinput() / screen.textinput() are turtle-specific GUI
# popups. numinput can also enforce a min/max right in the dialog.

# Uncomment to try in the terminal (note the int() conversion!):
# raw_answer = input("How many petals? ")
# print(type(raw_answer), raw_answer)        # <class 'str'> '12'
# petals_from_terminal = int(raw_answer)     # now it's a number

petals_requested = screen.numinput(
    "Petals", "How many petals? (3-36)", 12, minval=3, maxval=36
)
print("screen.numinput returned:", petals_requested, type(petals_requested))

favorite_color = screen.textinput("Color", "Name a color for the petals:")
print("screen.textinput returned:", favorite_color, type(favorite_color))

# Try clicking "Cancel" on one of the popups above and look at what
# gets printed -- that's the problem Section 7 solves.


# ---------------------------------------------------------------
# 7. Validation: don't trust the input blindly
# ---------------------------------------------------------------

# 7a. Validating text input against an allowed list (not a range).
allowed_colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]

if favorite_color is not None:
    favorite_color = favorite_color.strip().lower()  # tidy up what the user typed

if favorite_color in allowed_colors:
    chosen_color = favorite_color
    print(f"Using your color choice: {chosen_color}")
else:
    chosen_color = random.choice(allowed_colors)
    print(f"'{favorite_color}' isn't one of the allowed colors -- using {chosen_color} instead.")

# Common pitfall: forgetting .lower() on the input, so "Red" fails
# a check against "red" even though a person would call them the
# same thing.

# 7b. Simple version: fall back to a default if numeric input is bad.
if petals_requested is None or petals_requested < 3:
    petals_requested = 12
    print("Invalid or cancelled input -- using default of 12 petals.")

# 7c. Stronger version: keep asking until the answer is valid. This
# one is left ACTIVE (not commented out) so students see the dialog
# reappear immediately if they hit Cancel or an out-of-range value.
size_requested = None
while size_requested is None:
    size_requested = screen.numinput(
        "Size", "How big should each petal be? (20-100)", 60, minval=20, maxval=100
    )
print("Got a valid size:", size_requested)

t.penup()
t.goto(-150, -100)
t.pendown()
t.pencolor("black")
t.fillcolor(chosen_color)
t.begin_fill()
draw_polygon(t, sides=4, size=50)
t.end_fill()

t.penup()
t.goto(150, -100)
t.pendown()
draw_flower(t, petals=int(petals_requested), sides=5, size=size_requested, colors=petal_colors)

# Ask students to compare 7b and 7c out loud: 7b accepts one bad
# answer and quietly substitutes a default; 7c refuses to move on
# until it gets something usable. Neither is "more correct" --
# which one fits depends on whether a default is good enough.


# ---------------------------------------------------------------
# 8. OPTIONAL -- Labeling with write()
# ---------------------------------------------------------------
# write() drops text at the turtle's CURRENT position -- it's
# common to penup(), goto() where you want the label, then write().

t.penup()
t.goto(150, -170)
t.pendown()
t.color("black")
t.write(
    f"{int(petals_requested)} petals, size {int(size_requested)}",
    font=("Arial", 12, "normal"),
)


# ---------------------------------------------------------------
# 9. Random walk: a loop that picks something new each time
# ---------------------------------------------------------------
# Same "for loop calling code with a changing value" idea from
# Session 2 -- except now the value is random instead of a fixed
# pattern like `20 + i * 15`.

walker = turtle.Turtle()
walker.speed(0)
walker.color("black")
walker.pensize(2)
walker.penup()
walker.goto(-250, -50)
walker.pendown()

for _ in range(50):
    angle = random.randint(0, 360)
    distance = random.randint(10, 40)
    walker.setheading(angle)
    walker.forward(distance)

# Ask students to predict the shape of the path BEFORE running this
# section. Compare a few runs side by side -- no two paths match.


# 9b. Bonus: random walk that also changes color and thickness
# --------------------------------------------------------------
# Brings conditionals back into the picture: every 10 steps, pick a
# fresh random color and pen thickness, using the same i % 10 == 0
# idea from Session 2's conditionals section.

walker2 = turtle.Turtle()
walker2.speed(0)
walker2.penup()
walker2.goto(250, -50)
walker2.pendown()

walk_colors = ["red", "orange", "gold", "green", "blue", "purple"]

for i in range(60):
    if i % 10 == 0:                                  # every 10th step...
        walker2.color(random.choice(walk_colors))    # ...pick a new color
        walker2.pensize(random.randint(1, 4))         # ...and a new thickness
    angle = random.randint(0, 360)
    distance = random.randint(8, 25)
    walker2.setheading(angle)
    walker2.forward(distance)

# Talking point: this is the same random walk as 9a, with one
# conditional added. A good example of how small, familiar pieces
# (a loop, an if, a function call) combine into something that
# looks a lot more complex than any single piece on its own.


# ---------------------------------------------------------------
# Keep the window open until the user closes it
# ---------------------------------------------------------------
turtle.done()
