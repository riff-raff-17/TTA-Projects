"""
Rainbow Spiral Galaxy — Python Turtle Art
-------------------------------------------
Draws a single continuously growing line that spirals outward, turning
a fixed angle each step. Because the angle isn't a clean divisor of 360,
the arms fan out unevenly like a galaxy, and the color smoothly shifts
through the rainbow as the spiral grows.

Run it with: python spiral_galaxy.py
Click anywhere on the window to close it when it's done.
"""

import turtle
import colorsys


def draw_spiral_galaxy(steps=300, angle=59, growth=0.35):
    screen = turtle.Screen()
    screen.bgcolor("black")
    screen.title("Rainbow Spiral Galaxy")
    screen.tracer(1)  # instant drawing, we refresh manually at the end

    t = turtle.Turtle()
    t.speed(20)
    t.width(2)
    t.hideturtle()

    for i in range(steps):
        # Smoothly cycle the hue through the rainbow as the spiral grows
        hue = (i / steps) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        t.color(r, g, b)

        t.forward(i * growth)  # each segment is a little longer than the last
        t.left(angle)          # turning by a non-divisor of 360 fans out the arms

    screen.update()
    screen.exitonclick()


if __name__ == "__main__":
    draw_spiral_galaxy(steps=1000)