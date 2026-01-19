import turtle
import random

class Robot:
    def __init__(self, color):
        self.t = turtle.Turtle()
        self.t.color(color)
        self.t.shape("turtle")
        self.t.speed(1)
        self.energy = 10  # internal state you manage

    def step(self):
        if self.energy <= 0:
            return
        self.t.forward(random.randint(10, 30))
        self.t.left(random.randint(-60, 60))
        self.energy -= 0

bots = [Robot("red"), Robot("blue"), Robot("green"), Robot("purple"), Robot("brown")]
for _ in range(2000):
    for b in bots:
        b.step()

turtle.done()
