import turtle

# 1. Create the screen and the turtle

screen = turtle.Screen() # creates a window
screen.title("Turtle Fundamentals")

t = turtle.Turtle() # creates a turtle object
t.speed(1) # 0 = fastest, 1–10 = slower

# 2. Basic Movement

# Move forward 100 pixels
t.forward(100)

# Turn right 90 degrees
t.right(90)

# Move forward again
t.forward(100)

# Turn left 45 degrees
t.left(45)

# Move backward
t.backward(50)


# 3. Changing Pen Settings

t.pensize(3)           # thicker lines
t.color("blue")        # change pen color

t.forward(80)
t.right(90)
t.forward(80)

t.color("red")         # new color
t.pensize(5)
t.forward(80)

# 4. Pen Up / Pen Down
# Move the turtle without drawing

t.penup()
t.goto(-50, 50) # move to a new position
t.pendown()

t.color("green")
t.circle(40) # draw a circle with radius 40


# 5. Drawing with Loops

# Square using a loop
t.color("purple")
for _ in range(4):
    t.forward(60)
    t.right(90)

# Keeps the window open until the user closes it
turtle.done()
