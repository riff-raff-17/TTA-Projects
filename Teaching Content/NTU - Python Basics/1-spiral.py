'''
Try changing:

Turn angle (t.right(30))
- Try 10, 20, 45, 91 degrees

Colors
- Replace the palette or generate random RGB colors.

Growth rate (distance += 1)
- Increase faster, bigger, looser spiral
- Increase slower, tighter spiral

Number of loop iterations
'''

import turtle
import random

# --------------------------------------------------
# 1. Setup the screen and turtle

screen = turtle.Screen()
screen.bgcolor("black") # for contrast

t = turtle.Turtle()
t.speed(0) # fastest drawing
t.width(2) # line thickness


# Optional: define a set of nice colors
colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple"]

# 2. Spiral drawing loop

distance = 5 # how far the turtle moves each step

for angle in range(50): # try larger ranges for bigger spirals!
    t.color(random.choice(colors)) # pick a random color
    t.forward(distance)
    t.right(30) # experiment with different angles
    distance += 1 # slowly increase movement to make a spiral

turtle.done()
