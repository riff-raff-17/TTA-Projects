import turtle

# -----------------------------
# SETUP
# -----------------------------
screen = turtle.Screen()
screen.title("Session 1 - Meet the Turtle Object")

t = turtle.Turtle()  # <- This is our turtle OBJECT
t.shape("turtle")
t.speed(3)

# Teacher talk / prompt:
# "This 't' is an object. We can send it commands like t.forward(...)."
# Ask: "What do you think t.speed(3) does?"

# -----------------------------
# WARM-UP DEMO
# -----------------------------
t.forward(100)
t.left(90)
t.forward(80)

# Ask students:
# - "What direction is the turtle facing now?"
# - "Where is it on the screen?"
# - "Does the turtle remember what it already did?"

# -----------------------------
# EXERCISE 1: CHANGE STYLE
# -----------------------------
# Student task (2-5 minutes):
# 1) Change the pen color
# 2) Change the pen size
# 3) Change the turtle speed
#
# Try:
# t.color("red")
# t.pensize(5)
# t.speed(10)

# TODO: Uncomment and adjust these lines
# t.color("blue")
# t.pensize(4)

# -----------------------------
# EXERCISE 2: DRAW A SQUARE (GUIDED)
# -----------------------------
# Teacher prompt:
# "We can repeat steps with a loop. A square is 4 sides, 4 turns."

# TODO: Fill in the loop to draw a square.
# Hint: forward(100) and left(90)

# for _ in range(4):
#     t.forward(100)
#     t.left(90)

# Ask:
# - "Why 90 degrees?"
# - "What happens if we turn 120 degrees?"

# -----------------------------
# MINI-DEMO: TURTLE HAS STATE
# -----------------------------
# The turtle knows (state): its position and direction.
# We can ask it questions using methods that return values.

pos = t.position()      # current (x, y)
heading = t.heading()   # direction in degrees

print("Current position:", pos)
print("Current heading:", heading)

# Ask students:
# - "What do you notice about position?"
# - "If we run it again after moving, will these numbers change?"

# -----------------------------
# EXERCISE 3: MOVE TO A NEW START (UNGUIDED)
# -----------------------------
# Student task:
# Move the turtle to a new location without drawing a line, then draw something.
# Hints:
# t.penup()
# t.goto(x, y)
# t.pendown()

# TODO: Try moving to a new place:
# t.penup()
# t.goto(-150, 100)
# t.pendown()

# Challenge:
# - Draw a triangle (3 sides), then a square (4 sides), in different places.

# -----------------------------
# CLOSING QUESTIONS
# -----------------------------
# Ask:
# 1) "What is an object?"
# 2) "Name 3 methods we used today."
# 3) "What is something the turtle remembers (state)?"

screen.mainloop()
