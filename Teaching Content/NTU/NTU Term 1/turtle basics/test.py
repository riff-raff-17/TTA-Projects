import turtle

screen = turtle.Screen()
bob = turtle.Turtle()
screen.bgcolor("black")

bob.speed(0)

turtle.tracer(10, 0.5)


def crazy():
    for i in range(360):
        for colors in ['red', 'yellow', 'green', 'purple', 'orange', 'blue']:
            bob.pencolor(colors)
            bob.forward(i)
            bob.left(124)


crazy()
turtle.update()

turtle.done()
