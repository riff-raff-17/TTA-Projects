'''Simple implementation of tkinter gui. This can be expanded as needed.
Just a simple blueprint to get started.I've commented most of the tkinter 
functions as best I can so hopefully it's readable.'''

import tkinter as tk
import threading

# Don't import chatbot.py directly to avoid running its input‐loop on import!
# Also makes everyone have to rewrite the code so they really understand it
commands = {
    "exit": "Exits the chatbot",
    "help": "Displays the list of available commands.",
}

class ChatBotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ChatBot GUI")
        self.geometry("500x400")

        # This flag tracks whether the bot is waiting for the user's follow-up after "hello".
        self.awaiting_followup = False

        # Frame to hold the conversation (Text widget) and scrollbar
        text_frame = tk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        # Vertical scrollbar
        self.scrollbar = tk.Scrollbar(text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Read-only Text widget to display conversation
        self.text_area = tk.Text(
            text_frame,
            height=15,
            wrap=tk.WORD,
            yscrollcommand=self.scrollbar.set,
            state=tk.DISABLED
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_area.yview)

        # Entry widget for user input
        entry_frame = tk.Frame(self)
        entry_frame.pack(fill=tk.X, padx=10, pady=10)

        self.user_entry = tk.Entry(entry_frame)
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.user_entry.bind("<Return>", self.on_enter_pressed)

        send_button = tk.Button(entry_frame, text="Send", command=self.on_send_clicked)
        send_button.pack(side=tk.RIGHT)

        # Print initial greeting
        self._append_text("ChatBot: Hi! I am a ChatBot. What would you like to do?\n"
                          "Type 'help' for a list of commands.\n"
                          "Type 'exit' to exit the chatbot.\n\n")

    def _append_text(self, message: str):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, message)
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def on_enter_pressed(self, event):
        self._process_user_input()

    def on_send_clicked(self):
        self._process_user_input()

    def _process_user_input(self):
        user_text = self.user_entry.get().strip()
        if not user_text:
            return

        # Display “You: …” in the text area
        self._append_text(f"You: {user_text}\n")
        self.user_entry.delete(0, tk.END)

        low = user_text.lower()

        # If we were waiting for follow‐up after saying "Hello"
        if self.awaiting_followup:
            # Bot responds "I see!" regardless of what the user typed
            # You can change this to whatever you want, just a prototype
            threading.Timer(0.1, lambda: self._append_text("ChatBot: I see!\n\n")).start()
            self.awaiting_followup = False
            return

        # Handle commands
        if "exit" in low:
            # Bot says bye and then quits
            threading.Timer(0.1, lambda: self._append_text("ChatBot: See you!\n")).start()
            self.after(500, self.destroy)  # close window after a short delay
        elif "help" in low:
            # List all available commands
            response = "ChatBot: Here are the available commands:\n"
            for cmd, desc in commands.items():
                response += f"- {cmd}: {desc}\n"
            response += "\n"
            threading.Timer(0.1, lambda: self._append_text(response)).start()
        elif "hello" in low:
            # Greet and set awaiting_followup flag
            threading.Timer(0.1, lambda: self._append_text("ChatBot: Hello! How are you doing?\n")).start()
            self.awaiting_followup = True
        else:
            # Default fallback
            threading.Timer(0.1, lambda: self._append_text("ChatBot: I don't understand that command.\n\n")).start()


if __name__ == "__main__":
    app = ChatBotGUI()
    app.mainloop()
