#!/usr/bin/python3
import pathlib
import tkinter as tk
import pygubu
from testui import testingUI


class testing(testingUI):
    def __init__(self, master=None):
        super().__init__(master)


if __name__ == "__main__":
    app = testing()
    app.run()
