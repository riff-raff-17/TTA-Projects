'''Tkinter GUI ChatBot with extended features.
If using UGOT, add in the specific movement commands in the UGOT Control mode
and connect to the IP address at the start'''

import tkinter as tk
from tkinter import simpledialog, messagebox
import threading
import re
import random
from textblob import TextBlob
'''Uncomment this out and add in correct ip

from ugot import ugot
got = ugot.UGOT()
got.initialize('0.0.0.0')'''

# In-memory user store
users = {'rafa': 'password'}

# Descriptions for the help command
commands_desc = {
    "exit": "Exits the chatbot",
    "help": "Displays the list of available commands",
    "calc": "Enter calculator mode",
    "game": "Play a number guessing game",
    "ugot": "Control the UGOT robot",
}

class ChatBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChatBot GUI")
        self.geometry("600x400")

        # Frames for login and chat
        self.login_frame = tk.Frame(self)
        self.chat_frame = tk.Frame(self)

        self._build_login_frame()
        self._build_chat_frame()

        self.login_frame.pack(fill=tk.BOTH, expand=True)

        # Chat state
        self.mode = None  # None, 'calculator', 'game', or 'ugot'
        self.secret_number = None
        self.attempts = 0

    def _build_login_frame(self):
        tk.Label(self.login_frame, text="Username:").pack(pady=5)
        self.username_entry = tk.Entry(self.login_frame)
        self.username_entry.pack(pady=5)
        tk.Label(self.login_frame, text="Password:").pack(pady=5)
        self.password_entry = tk.Entry(self.login_frame, show="*")
        self.password_entry.pack(pady=5)

        btn_frame = tk.Frame(self.login_frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Login", command=self._handle_login).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Sign Up", command=self._handle_signup).pack(side=tk.LEFT, padx=5)

        self.login_message = tk.Label(self.login_frame, text="", fg="red")
        self.login_message.pack()

    def _build_chat_frame(self):
        self.text_area = tk.Text(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        entry_frame = tk.Frame(self.chat_frame)
        entry_frame.pack(fill=tk.X, padx=10, pady=5)
        self.user_entry = tk.Entry(entry_frame)
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_entry.bind("<Return>", lambda e: self._process_user_input())
        tk.Button(entry_frame, text="Send", command=self._process_user_input).pack(side=tk.RIGHT)

    def _append_text(self, message: str):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message)
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def _handle_login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()
        if username in users and users[username] == password:
            self.login_message.config(text="", fg="green")
            self._show_chat()
        else:
            self.login_message.config(text="Invalid username or password", fg="red")

    def _handle_signup(self):
        new_username = simpledialog.askstring("Sign Up", "Enter new username:")
        if not new_username:
            return
        if new_username in users:
            messagebox.showerror("Error", "User already exists!")
            return
        new_password = simpledialog.askstring("Sign Up", "Enter new password:", show='*')
        if not new_password:
            return
        users[new_username] = new_password
        messagebox.showinfo("Success", "User added successfully!")

    def _show_chat(self):
        self.login_frame.pack_forget()
        self.chat_frame.pack(fill=tk.BOTH, expand=True)
        self._append_text(
            "ChatBot: Hi! I am a ChatBot. What would you like to do?\n"
            "Type 'help' for a list of commands.\n"
            "Type 'exit' to exit the chatbot.\n\n"
        )

    def _process_user_input(self):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return
        self._append_text(f"You: {user_text}\n")
        self.user_entry.delete(0, tk.END)
        low = user_text.lower()

        # Calculator mode
        if self.mode == 'calculator':
            if low == 'exit':
                self.mode = None
                self._append_text("ChatBot: Exiting calculator mode.\n\n")
            else:
                m = re.match(r'(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)', user_text)
                if m:
                    x, op, y = float(m.group(1)), m.group(2), float(m.group(3))
                    ops = {
                        '+': x + y,
                        '-': x - y,
                        '*': x * y,
                        '/': x / y if y != 0 else "Error: Division by zero"
                    }
                    result = ops.get(op)
                    self._append_text(f"ChatBot: {result}\n\n")
                else:
                    self._append_text("ChatBot: Invalid expression. Please enter in format 'a + b'.\n\n")
            return

        # Number guessing game mode
        if self.mode == 'game':
            if low == 'exit':
                self.mode = None
                self._append_text("ChatBot: Exiting game mode.\n\n")
            else:
                try:
                    guess = int(user_text)
                except ValueError:
                    self._append_text("ChatBot: Please enter a valid number.\n\n")
                    return
                self.attempts += 1
                if guess == self.secret_number:
                    self._append_text(f"ChatBot: Congratulations! You guessed it in {self.attempts} attempts.\n\n")
                    self.mode = None
                elif guess < self.secret_number:
                    self._append_text("ChatBot: Too low.\n\n")
                else:
                    self._append_text("ChatBot: Too high.\n\n")
            return

        # UGOT control mode
        '''Add in correct movement commands for UGOT here'''
        if self.mode == 'ugot':
            if low == 'exit':
                self.mode = None
                self._append_text("ChatBot: Exiting UGOT mode.\n\n")
            else:
                # Spell-check the command
                corrected = TextBlob(user_text).correct()
                if str(corrected) != user_text:
                    self._append_text(f"ChatBot: Did you mean: '{corrected}'?\n")
                    command = str(corrected)
                else:
                    command = user_text
                parts = command.lower().split()
                for i in range(0, len(parts), 2):
                    cmd = parts[i]
                    if i + 1 < len(parts):
                        try:
                            val = int(parts[i+1])
                        except ValueError:
                            continue
                        if cmd == 'forward':
                            self._append_text(f"ChatBot: UGOT moving forward {val} units.\n")
                        elif cmd == 'backward':
                            self._append_text(f"ChatBot: UGOT moving backward {val} units.\n")
                        elif cmd == 'left':
                            self._append_text(f"ChatBot: UGOT turning left {val} degrees.\n")
                        elif cmd == 'right':
                            self._append_text(f"ChatBot: UGOT turning right {val} degrees.\n")
                self._append_text("\n")
            return

        # No active mode, handle top-level commands
        if "exit" in low:
            self._append_text("ChatBot: See you!\n")
            self.after(500, self.destroy)

        elif "help" in low:
            resp = "ChatBot: Available commands:\n"
            for cmd, desc in commands_desc.items():
                resp += f"- {cmd}: {desc}\n"
            resp += "\n"
            self._append_text(resp)

        elif low.startswith("calc"):
            self.mode = 'calculator'
            self._append_text("ChatBot: Entering calculator mode. Type 'exit' to return.\n")

        elif low.startswith("game"):
            self.mode = 'game'
            self.secret_number = random.randint(1, 100)
            self.attempts = 0
            self._append_text("ChatBot: I've picked a number between 1 and 100. Can you guess it?\n")

        elif low.startswith("ugot"):
            self.mode = 'ugot'
            self._append_text("ChatBot: Enter UGOT commands (e.g., 'forward 10 backward 5'), or 'exit' to leave.\n\n")

        elif low.startswith("validate"):
            cmdstr = user_text[len("validate"):].strip()
            ok = bool(re.match(r"^(?:\s*(forward|backward|left|right)\s+\d+\s*)+$", cmdstr))
            if ok:
                self._append_text(f"ChatBot: '{cmdstr}' is valid.\n\n")
            else:
                self._append_text(f"ChatBot: '{cmdstr}' is not valid.\n\n")
        else:
            self._append_text("ChatBot: I don't understand that command.\n\n")

if __name__ == "__main__":
    app = ChatBotGUI()
    app.mainloop()
