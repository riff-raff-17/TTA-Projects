"""
Mini Project: Polygon Flower
-----------------------------
Combines everything from this session -- functions, parameters,
loops, lists, and conditionals -- into one small generative drawing.
"""

import random
import turtle

screen = turtle.Screen()
screen.title("Polygon Flower")
screen.bgcolor("white")

t = turtle.Turtle()
t.speed(0)
t.hideturtle()


def draw_polygon(turtle_obj, sides, size):
    """Draw a regular polygon with the given number of sides and side length."""
    angle = 360 / sides
    for _ in range(sides):
        turtle_obj.forward(size)
        turtle_obj.right(angle)


def draw_flower(turtle_obj, petals, sides, size, colors):
    """
    Draw 'petals' copies of a polygon, each rotated evenly around
    a full circle, cycling through the given list of colors.
    """
    turn_between_petals = 360 / petals
    for i in range(petals):
        color = colors[i % len(colors)]   # cycle through the list
        turtle_obj.color(color)
        draw_polygon(turtle_obj, sides, size)
        turtle_obj.right(turn_between_petals)


# --- Draw the flower ---
petal_colors = ["red", "orange", "gold", "purple", "deeppink", "blue"]
draw_flower(t, petals=12, sides=6, size=80, colors=petal_colors)


# ---------------------------------------------------------------
# Challenge extensions -- try uncommenting one at a time
# ---------------------------------------------------------------

# 1. Random colors instead of a fixed list:
# def random_color():
#     return (random.random(), random.random(), random.random())
# screen.colormode(1.0)  # tells turtle to expect 0-1 floats, not 0-255 ints

# 2. A second, smaller flower drawn at a different position:
# t.penup()
# t.goto(200, 0)
# t.pendown()
# draw_flower(t, petals=8, sides=4, size=40, colors=["black", "gray"])

# 3. Ask the user for input instead of hardcoding petal count:
# num_petals = int(screen.numinput("Petals", "How many petals?", 12, minval=3, maxval=36))
# draw_flower(t, petals=num_petals, sides=6, size=80, colors=petal_colors)


turtle.done()
