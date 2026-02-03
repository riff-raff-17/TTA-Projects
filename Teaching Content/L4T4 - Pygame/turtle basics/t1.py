import turtle

class Pen:
    def __init__(self, color="black", speed=1):
        self.t = turtle.Turtle()
        self.t.color(color)
        self.t.speed(speed)

    def square(self, size):
        for _ in range(4):
            self.t.forward(size)
            self.t.left(90)

pen = Pen(color="purple")
pen.square(120)
turtle.done()
