
import turtle
import random

# 1. Screen and turtle setup

screen = turtle.Screen()
screen.bgcolor("black")

t = turtle.Turtle()
t.speed(0) # fastest speed
t.width(2) # line thickness
t.hideturtle() # hide the turtle icon

colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "white"]


# 2. Draw one spirograph ring

def draw_spirograph(radius=100, step_angle=5):
    """
    Draw a spirograph ring by drawing many circles,
    rotating a little bit between each one.

    radius  radius of each circle
    step_angle : how many degrees to rotate after each circle
                (smaller values = more circles = denser pattern)
    """
    # Number of circles needed to complete (approximately) a full rotation.
    # Example: 360 / 5 = 72 circles.
    num_circles = int(360 / step_angle)

    for _ in range(num_circles):
        t.color(random.choice(colors))
        t.circle(radius)
        t.right(step_angle)


# 3. Draw multiple spirograph rings

# Center at origin
t.penup()
t.goto(0, 0)
t.pendown()

# You can experiment with these sets of parameters
spiro_configs = [
    (80, 5),
    (120, 10),
    (60, 7),
]

for radius, step in spiro_configs:
    draw_spirograph(radius=radius, step_angle=step)
    # Slight rotation between whole rings for extra effect
    t.right(10)

turtle.done()
