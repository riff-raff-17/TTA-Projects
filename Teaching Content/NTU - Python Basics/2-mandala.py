import turtle
import random

# 1. Screen and turtle setup

screen = turtle.Screen()
screen.bgcolor("black")

t = turtle.Turtle()
t.speed(0) # fastest
t.width(2) # line thickness
t.hideturtle() # hide the turtle icon for cleaner look

colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta"]

# 2. Draw a regular polygon

def draw_polygon(side_length, sides):
    """Draw a regular polygon with given side length and number of sides"""
    angle = 360 / sides
    for _ in range(sides):
        t.forward(side_length)
        t.right(angle)

# 3. Draw the mandala

# Try changing these values:
num_petals = 36 # how many shapes around the circle
sides = 6 # how many sides per polygon (e.g., 3, 4, 5, 6, 8)
side_length = 80 # size of each polygon
rotation_angle = 360 / num_petals

for _ in range(num_petals):
    t.color(random.choice(colors))

    # With polygons
    draw_polygon(side_length, sides)

    # Alternatively, with circles
    # t.circle(side_length)

    t.right(rotation_angle)

turtle.done()
