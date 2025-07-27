#!/usr/bin/python3
import pathlib
import tkinter as tk
import pygubu

PROJECT_PATH = pathlib.Path(__file__).parent
PROJECT_UI = PROJECT_PATH / "app_ui.ui"
RESOURCE_PATHS = [PROJECT_PATH]


class testingUI:
    def __init__(
        self,
        master=None,
        translator=None,
        on_first_object_cb=None,
        data_pool=None
    ):
        self.builder = pygubu.Builder(
            translator=translator,
            on_first_object=on_first_object_cb,
            data_pool=data_pool
        )
        self.builder.add_resource_paths(RESOURCE_PATHS)
        self.builder.add_from_file(PROJECT_UI)
        # Main widget
        self.mainwindow: tk.Toplevel = self.builder.get_object(
            "toplevel1", master)
        self.entry_input: tk.Entry = self.builder.get_object(
            "entry_input", master)
        self.builder.connect_callbacks(self)

    def run(self):
        self.mainwindow.mainloop()

    def on_clicked(self):
        user_text = self.entry_input.get()
        print("You entered:", user_text)

    def on_checked(self):
        pass


if __name__ == "__main__":
    app = testingUI()
    app.run()
