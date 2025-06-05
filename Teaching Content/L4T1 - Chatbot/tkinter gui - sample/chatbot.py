#Dictionary to hold all the commands
commands = {
    "exit": "Exits the chatbot",
    "help": "Displays the list of available commands.",
}

print("Hi! I am a ChatBot. What would you like to do? \nType \'help\' for a list of commands. \
\nType \'exit\' to exit the chatbot.'")

while True:
    prompt = input("You: ")
    low = prompt.lower()
    
    if 'exit' in low:
        print("See you!")
        break
        
    elif 'help' in low:
        print("\nChatBot: Here are the available commands:")
        for command, description in commands.items():
            print(f"- {command}: {description}")
            
    elif 'hello' in low:
        print("Hello! How are you doing?")
        input("You: ")
        print("I see!")
        
    else:
        print("I don't understand that command.")