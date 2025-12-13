import turtle
import random

# 1. Screen and turtle setup

screen = turtle.Screen()
screen.bgcolor("black") # night sky

t = turtle.Turtle()
t.speed(0) # fastest
t.width(2)
t.hideturtle()

colors = ["red", "orange", "yellow", "green", "cyan", "blue", "magenta", "white"]


# 2. Draw a single firework burst

def draw_firework(x, y, radius=100, rays=36):
    """
    Draw a single firework centered at (x, y).
    
    x, y   : center position
    radius : length of each ray
    rays   : how many rays (lines) in the burst
    """
    t.penup()
    t.goto(x, y)
    t.pendown()
    
    t.color(random.choice(colors))
    
    angle = 360 / rays
    
    for _ in range(rays):
        t.forward(radius)
        t.backward(radius)
        t.right(angle)


# 3. Draw multiple random fireworks

num_fireworks = 8 # try changing this!

for _ in range(num_fireworks):
    # Pick a random position for each firework
    x = random.randint(-250, 250)
    y = random.randint(-150, 200)
    
    # Random size and number of rays
    radius = random.randint(50, 130)
    rays = random.choice([16, 24, 32, 40])
    
    draw_firework(x, y, radius, rays)

turtle.done()
