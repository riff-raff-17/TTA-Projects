"""
Session 2: Variables, Functions, Loops, Lists & Conditionals with Turtle
--------------------------------------------------------------------------
Builds directly on turtle_basics.py from last session.

TEACHING TIP: Don't run this whole file at once on the first pass.
Build it up section by section live, running after each new section
is added, so students see the drawing grow. The full file is included
here so you (and students) have a working reference afterward.
"""

import turtle

# ---------------------------------------------------------------
# 0. Setup (same idea as last time)
# ---------------------------------------------------------------

screen = turtle.Screen()
screen.title("Functions, Loops, and Lists")

t = turtle.Turtle()
t.speed(0)  # fastest -- we're drawing a lot more this time


# ---------------------------------------------------------------
# 1. Variables: naming the "magic numbers"
# ---------------------------------------------------------------
# Last time we typed forward(100), right(90), etc. directly into
# the code. Storing these as variables makes the code easier to
# read, and easier to change in one place.

side_length = 60
turn_angle = 90

for _ in range(4):
    t.forward(side_length)
    t.right(turn_angle)

# Try it live: change side_length and turn_angle and re-run.
# Ask students: what shape do you get with turn_angle = 120? Why?


# ---------------------------------------------------------------
# 2. Functions: packaging up behavior so we can reuse it
# ---------------------------------------------------------------
# A function lets us name a chunk of code and reuse it with
# different inputs (parameters), instead of copy-pasting.


def draw_square(turtle_obj, size):
    """Draw a square of the given size using the given turtle."""
    for _ in range(4):
        turtle_obj.forward(size)
        turtle_obj.right(90)


t.penup()
t.goto(-150, 100)
t.pendown()
draw_square(t, 40)

t.penup()
t.goto(-50, 100)
t.pendown()
draw_square(t, 80)

# turtle_obj and size are PARAMETERS: placeholders that get filled
# in with real values (ARGUMENTS) each time we CALL the function.


# 2b. A second function, same pattern
# ------------------------------------
# Once draw_square exists, draw_triangle is a great "you try it"
# moment -- same shape (turtle_obj, size), same idea, different
# angle and range. Live-code this WITH students, or hide it and
# have them write it themselves first.


def draw_triangle(turtle_obj, size):
    """Draw an equilateral triangle of the given size using the given turtle."""
    for _ in range(3):
        turtle_obj.forward(size)
        turtle_obj.right(120)  # 360 / 3 sides = 120 degrees per turn


t.penup()
t.goto(50, 100)
t.pendown()
draw_triangle(t, 60)

t.penup()
t.goto(150, 100)
t.pendown()
draw_triangle(t, 100)

# Ask students: draw_square turns 90 degrees, draw_triangle turns
# 120. Where does that number come from? (360 / number of sides --
# this is the exact idea the mini-project generalizes into
# draw_polygon later.)


# ---------------------------------------------------------------
# 3. Nested loops: calling a function repeatedly, in a pattern
# ---------------------------------------------------------------
# A loop that calls our function is much shorter than five
# copy-pasted blocks -- and it's easy to change "how many".

t.penup()
t.goto(-150, -50)
t.pendown()

for i in range(5):
    size = 20 + i * 15  # grows: 20, 35, 50, 65, 80
    draw_square(t, size)
    t.penup()
    t.forward(size + 20)  # move over before the next square
    t.pendown()


# 3b. A TRULY nested loop (bonus / stretch)
# -------------------------------------------
# The loop above calls the function repeatedly, but it's still
# just one loop. A "nested loop" is a loop INSIDE another loop --
# useful whenever you want rows AND columns, like a grid.
#
# Point out: the outer loop runs once per ROW; the inner loop runs
# completely (all 4 squares) before the outer loop moves on to the
# next row. Ask students to predict how many squares get drawn in
# total before running it (3 rows x 4 columns = 12).

t.penup()
t.goto(-150, -250)
t.pendown()

rows = 3
cols = 4
grid_start_x, grid_start_y = -150, -250

for row in range(rows):
    for col in range(cols):
        t.penup()
        t.goto(grid_start_x + col * 40, grid_start_y - row * 40)
        t.pendown()
        draw_square(t, 25)


# ---------------------------------------------------------------
# 4. Lists: a container of related values
# ---------------------------------------------------------------
# A list stores multiple values in one variable. We can loop over
# it directly with "for x in my_list".

colors = ["red", "blue", "green", "purple", "orange"]


def draw_colored_square(turtle_obj, size, color):
    turtle_obj.color(color)
    draw_square(turtle_obj, size)


t.penup()
t.goto(-150, -150)
t.pendown()

for color in colors:
    draw_colored_square(t, 30, color)
    t.penup()
    t.forward(50)
    t.pendown()

# 4b. Same result, looping by INDEX instead of by value
# ---------------------------------------------------------
# "for color in colors" hands you each VALUE directly.
# "for i in range(len(colors))" hands you each POSITION, and you
# look the value up yourself with colors[i]. Same output here --
# but the index version is what you need once you also want to
# know *where* you are in the list, which is exactly what the
# conditionals section (and colors[i % len(colors)] in the mini
# project) does next.

t.penup()
t.goto(-150, -220)
t.pendown()

for i in range(len(colors)):
    draw_colored_square(t, 30, colors[i])
    t.penup()
    t.forward(50)
    t.pendown()


# ---------------------------------------------------------------
# 5. Conditionals: making decisions in code
# ---------------------------------------------------------------
# if / else lets the program behave differently depending on a
# condition -- here, whether the loop counter is even or odd.

# 5a. Warm up on % (modulo) BEFORE bringing in turtle at all.
# --------------------------------------------------------------
# % gives the remainder after division. Read the prints below in
# the terminal first, then connect it to "even or odd" (a number
# is even exactly when there's nothing left over after dividing
# by 2 -- remainder 0).

print("5 % 2 =", 5 % 2)  # 1 -- 5 divided by 2 is 2 remainder 1
print("4 % 2 =", 4 % 2)  # 0 -- 4 divides evenly by 2
print("7 % 3 =", 7 % 3)  # 1 -- 7 divided by 3 is 2 remainder 1

t.penup()
t.goto(150, -100)
t.pendown()

for i in range(6):
    if i % 2 == 0:  # even index
        t.color("black")
    else:  # odd index
        t.color("gray")
    draw_square(t, 25)
    t.penup()
    t.forward(35)
    t.pendown()


# 5b. Bonus: elif for more than two outcomes
# ----------------------------------------------
# if / else only gives two branches. elif ("else if") lets you
# check additional conditions in order -- here, three colors based
# on remainder after dividing by 3 instead of 2.

t.penup()
t.goto(150, -170)
t.pendown()

for i in range(6):
    if i % 3 == 0:
        t.color("firebrick")
    elif i % 3 == 1:
        t.color("goldenrod")
    else:  # anything left over must be remainder 2
        t.color("forestgreen")
    draw_square(t, 25)
    t.penup()
    t.forward(35)
    t.pendown()


# ---------------------------------------------------------------
# Instructor answer key -- take-home challenges (uncomment to check)
# ---------------------------------------------------------------
# Don't hand these out with the file -- paste them in separately,
# or reveal after students have had a real attempt.

# --- Spiral challenge ---
# t.penup()
# t.goto(-250, -250)
# t.setheading(0)
# t.pendown()
# t.color("black")
# spiral_size = 5
# for _ in range(80):
#     t.forward(spiral_size)
#     t.right(4)
#     spiral_size += 2

# --- draw_star challenge ---
# def draw_star(turtle_obj, size):
#     """Draw a five-pointed star of the given size using the given turtle."""
#     for _ in range(5):
#         turtle_obj.forward(size)
#         turtle_obj.right(144)
#
# t.penup()
# t.goto(-250, 150)
# t.pendown()
# t.color("gold")
# draw_star(t, 80)
# t.penup()
# t.goto(-250, 30)
# t.pendown()
# draw_square(t, 40)   # two functions, one picture


# ---------------------------------------------------------------
# Keep the window open until the user closes it
# ---------------------------------------------------------------
turtle.done()
