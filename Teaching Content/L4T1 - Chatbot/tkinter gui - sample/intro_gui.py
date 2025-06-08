"""
This is a basic example to practice Tkinter. It covers:
- Window creation and geometry management
- A Canvas banner
- Menus (File and Help)
- Layout with pack() and grid()
- Entry widget with StringVar and event binding
- Buttons for adding, removing, and clearing tasks
- Listbox with Scrollbar to display items
- Messageboxes and File dialogs for user interaction
- Object-oriented design with a TodoApp class

Good way for the students to practice tkinter and OOP. Before getting into the
programming, go over classes with them to remind them how they work 
(or teach if they haven't done previously)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class TodoApp:
    """
    Main application class for the To-Do List.
    """
    def __init__(self, root):
        """
        Initialize the GUI, set up widgets and layout.
        """
        self.root = root
        root.title("To-Do List App")   # Window title
        root.geometry("400x500")        # Window size

        # Canvas Banner: displays the title in a custom banner
        banner = tk.Canvas(root, height=50, bg='#f0f0f0')
        banner.pack(fill='x')
        banner.create_text(200, 25, text="To-Do List App", font=("Helvetica", 16))

        # Create the application menu (File & Help)
        self.create_menu()

        # Entry Frame: for new task input and the Add button
        entry_frame = ttk.Frame(root)
        entry_frame.pack(padx=10, pady=10, fill='x')
        self.task_var = tk.StringVar()  # Holds input text
        task_entry = ttk.Entry(entry_frame, textvariable=self.task_var)
        task_entry.grid(row=0, column=0, sticky='ew')
        task_entry.bind('<Return>', lambda e: self.add_task())  # Enter key binding
        add_button = ttk.Button(entry_frame, text="Add Task", command=self.add_task)
        add_button.grid(row=0, column=1, padx=(5,0))
        entry_frame.columnconfigure(0, weight=1)  # Make entry expand

        # List Frame: contains the Listbox and its Scrollbar
        list_frame = ttk.Frame(root)
        list_frame.pack(padx=10, pady=(0,10), fill='both', expand=True)
        self.tasks_listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE)
        self.tasks_listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.tasks_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.tasks_listbox.config(yscrollcommand=scrollbar.set)
        list_frame.rowconfigure(0, weight=1)    # Make list expand vertically
        list_frame.columnconfigure(0, weight=1) # Make list expand horizontally

        # Button Frame: Remove Selected & Clear All buttons
        btn_frame = ttk.Frame(root)
        btn_frame.pack(padx=10, pady=(0,10))
        remove_button = ttk.Button(btn_frame, text="Remove Selected", command=self.remove_task)
        remove_button.pack(side='left', padx=(0,5))
        clear_button = ttk.Button(btn_frame, text="Clear All", command=self.clear_tasks)
        clear_button.pack(side='left')

    def create_menu(self):
        """
        Set up the menubar with File and Help menus.
        """
        menubar = tk.Menu(self.root)

        # File menu: Save, Load, Exit
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Tasks", command=self.save_tasks)
        file_menu.add_command(label="Load Tasks", command=self.load_tasks)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Help menu: About dialog
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # Attach menubar to window
        self.root.config(menu=menubar)

    def add_task(self):
        """
        Add a new task to the listbox. Warn if the entry is empty.
        """
        task = self.task_var.get().strip()
        if not task:
            messagebox.showwarning("Warning", "You must enter a task.")
            return
        self.tasks_listbox.insert('end', task)
        self.task_var.set("")  # Clear entry field

    def remove_task(self):
        """
        Remove all selected tasks from the listbox.
        """
        selected_indices = list(self.tasks_listbox.curselection())
        if not selected_indices:
            return
        # Delete in reverse order to avoid index shifting
        for index in reversed(selected_indices):
            self.tasks_listbox.delete(index)

    def clear_tasks(self):
        """
        Clear all tasks after user confirmation.
        """
        if messagebox.askyesno("Confirm", "Delete all tasks?"):
            self.tasks_listbox.delete(0, 'end')

    def save_tasks(self):
        """
        Save current tasks to a text file chosen by the user.
        """
        tasks = self.tasks_listbox.get(0, 'end')
        if not tasks:
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                for task in tasks:
                    f.write(task + "\n")

    def load_tasks(self):
        """
        Load tasks from a user-selected text file, replacing current list.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt")]
        )
        if file_path:
            with open(file_path, 'r') as f:
                tasks = f.readlines()
            self.tasks_listbox.delete(0, 'end')
            for task in tasks:
                self.tasks_listbox.insert('end', task.strip())

    def show_about(self):
        """
        Display an About messagebox.
        """
        messagebox.showinfo(
            "About",
            "Simple To-Do List App\nBuilt with Tkinter for practice purposes"
        )

if __name__ == "__main__":
    # Create the main application window and run the app
    root = tk.Tk()
    app = TodoApp(root)
    root.mainloop()
