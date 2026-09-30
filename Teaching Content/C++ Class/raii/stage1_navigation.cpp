// stage1_navigation.cpp
// Section 2: Building the World — Multi-Room Map & Navigation
// Compile: g++ -std=c++17 -o stage1_navigation stage1_navigation.cpp

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>

using namespace std;

class Room {
private:
    string name;
    string description;
    map<string, Room*> exits; // raw pointer for now — deliberately not the final version

public:
    Room(string n, string desc) : name(move(n)), description(move(desc)) {}

    void setExit(const string& direction, Room* room) { exits[direction] = room; }

    Room* getExit(const string& direction) const {
        auto it = exits.find(direction);
        return it == exits.end() ? nullptr : it->second;
    }

    void describe() const {
        cout << "\n== " << name << " ==" << endl;
        cout << description << endl;
        if (!exits.empty()) {
            cout << "Exits:";
            for (const auto& [dir, room] : exits) cout << " " << dir;
            cout << endl;
        }
    }

    const string& getName() const { return name; }
};

int main() {
    cout << "=== STAGE 1: NAVIGATION DEMO ===" << endl;

    // main() owns every room, via unique_ptr. The raw Room* pointers handed
    // out via setExit()/getExit() are safe because this vector outlives them.
    vector<unique_ptr<Room>> rooms;
    rooms.push_back(make_unique<Room>("Entrance Hall", "A dusty stone hall. Torches flicker on the walls."));
    rooms.push_back(make_unique<Room>("Armory", "Racks of rusted weapons line the walls."));
    rooms.push_back(make_unique<Room>("Damp Cave", "Water drips somewhere in the darkness."));

    rooms[0]->setExit("north", rooms[1].get());
    rooms[1]->setExit("south", rooms[0].get());
    rooms[1]->setExit("east", rooms[2].get());
    rooms[2]->setExit("west", rooms[1].get());

    Room* current = rooms[0].get();
    current->describe();
    cout << "\nCommands: look, go <direction>, quit\n";

    string line;
    while (true) {
        cout << "\n> ";
        if (!getline(cin, line)) break;

        istringstream iss(line);
        string cmd;
        iss >> cmd;
        string rest;
        getline(iss, rest);
        if (!rest.empty() && rest.front() == ' ') rest.erase(0, 1);

        if (cmd == "quit") {
            break;
        } else if (cmd == "look") {
            current->describe();
        } else if (cmd == "go") {
            Room* next = current->getExit(rest);
            if (!next) {
                cout << "You can't go that way." << endl;
            } else {
                current = next;
                cout << "You head " << rest << "." << endl;
                current->describe();
            }
        } else {
            cout << "Unknown command." << endl;
        }
    }

    return 0;
}
