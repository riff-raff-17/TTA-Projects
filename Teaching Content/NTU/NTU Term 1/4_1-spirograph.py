"""
Colorful Spirograph — Python Turtle Art
----------------------------------------
Draws many overlapping circles, each rotated slightly from the last,
cycling through the rainbow as it goes. The overlapping arcs create a
hypnotic, flower/mandala-like pattern.

Run it with: python spirograph.py
Click anywhere on the window to close it when it's done.
"""

import turtle
import colorsys


def draw_spirograph(n=200, radius=150):
    screen = turtle.Screen()
    screen.bgcolor("black")
    screen.title("Colorful Spirograph")
    screen.tracer(1)  # turn off auto-refresh so drawing is instant, then we update manually

    t = turtle.Turtle()
    t.speed(15)
    t.width(2)
    t.hideturtle()

    for i in range(n):
        # Cycle the hue smoothly through the rainbow (0.0 -> 1.0)
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        t.color(r, g, b)

        t.circle(radius)
        t.left(360 / n)  # rotate slightly before drawing the next circle

    screen.update()
    screen.exitonclick()


if __name__ == "__main__":
    draw_spirograph()