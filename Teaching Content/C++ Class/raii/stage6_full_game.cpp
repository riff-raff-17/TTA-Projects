// stage6_full_game.cpp
// Section 8: Full Game Loop Integration — the finished game.
// Builds on Stage 5 by adding the help command, win condition, and final report.
// Compile:  g++ -std=c++17 -o dungeon_crawler dungeon_crawler.cpp
// Run:      ./dungeon_crawler

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <numeric>

using namespace std;

class Player; // forward declaration — Item::use() needs a reference to Player

// ============================================================
// ITEMS — unique_ptr-owned, polymorphic, RAII-managed
// ============================================================

class Item {
protected:
    string name;
    int value;
public:
    Item(string n, int v) : name(move(n)), value(v) {}
    virtual ~Item() { cout << "  [" << name << " destroyed.]" << endl; }
    virtual void use(Player& player) = 0;
    virtual string describe() const = 0;
    const string& getName() const { return name; }
    int getValue() const { return value; }
};

class Potion : public Item {
    int healAmount;
public:
    Potion(string n, int v, int heal) : Item(move(n), v), healAmount(heal) {}
    void use(Player& player) override; // defined after Player is complete
    string describe() const override {
        ostringstream oss;
        oss << name << " (Potion, heals " << healAmount << ", worth " << value << ")";
        return oss.str();
    }
};

class Weapon : public Item {
    int damage;
public:
    Weapon(string n, int v, int dmg) : Item(move(n), v), damage(dmg) {}
    void use(Player& player) override; // equips itself
    int getDamage() const { return damage; }
    string describe() const override {
        ostringstream oss;
        oss << name << " (Weapon, damage " << damage << ", worth " << value << ")";
        return oss.str();
    }
};

class Armor : public Item {
    int defense;
public:
    Armor(string n, int v, int def) : Item(move(n), v), defense(def) {}
    void use(Player& player) override; // equips itself
    int getDefense() const { return defense; }
    string describe() const override {
        ostringstream oss;
        oss << name << " (Armor, defense " << defense << ", worth " << value << ")";
        return oss.str();
    }
};

// ============================================================
// ENEMIES — unique_ptr-owned by Room, polymorphic combat behavior
// ============================================================

class Enemy {
protected:
    string name;
    int health;
    int attackPower;
public:
    Enemy(string n, int hp, int atk) : name(move(n)), health(hp), attackPower(atk) {}
    virtual ~Enemy() { cout << "  [" << name << " is gone.]" << endl; }

    bool isAlive() const { return health > 0; }

    void takeDamage(int dmg) {
        health -= dmg;
        if (health < 0) health = 0;
        cout << "  " << name << " takes " << dmg << " damage (" << health << " HP left)." << endl;
    }

    virtual int attack() const { return attackPower; }

    virtual string describe() const {
        ostringstream oss;
        oss << name << " (HP: " << health << ", ATK: " << attackPower << ")";
        return oss.str();
    }

    const string& getName() const { return name; }
    int getHealth() const { return health; }
};

class Goblin : public Enemy {
public:
    explicit Goblin(string n = "Goblin") : Enemy(move(n), 20, 4) {}
};

class Dragon : public Enemy {
public:
    explicit Dragon(string n = "Dragon") : Enemy(move(n), 60, 12) {}
    int attack() const override {
        cout << "  " << name << " breathes fire!" << endl;
        return attackPower;
    }
};

// ============================================================
// PLAYER
// ============================================================

class Player {
private:
    string name;
    int health;
    int maxHealth;
    vector<unique_ptr<Item>> inventory;
    Weapon* equippedWeapon = nullptr; // non-owning observer into inventory
    Armor* equippedArmor = nullptr;   // non-owning observer into inventory

public:
    explicit Player(string n, int hp = 100) : name(move(n)), health(hp), maxHealth(hp) {}

    void pickUp(unique_ptr<Item> item) {
        cout << "Picked up " << item->getName() << "." << endl;
        inventory.push_back(move(item));
    }

    Item* findItem(const string& itemName) {
        auto it = find_if(inventory.begin(), inventory.end(),
            [&](const unique_ptr<Item>& i) { return i->getName() == itemName; });
        return it == inventory.end() ? nullptr : it->get();
    }

    unique_ptr<Item> removeItem(const string& itemName) {
        auto it = find_if(inventory.begin(), inventory.end(),
            [&](const unique_ptr<Item>& i) { return i->getName() == itemName; });
        if (it == inventory.end()) return nullptr;
        unique_ptr<Item> found = move(*it);
        inventory.erase(it);
        if (equippedWeapon == found.get()) equippedWeapon = nullptr;
        if (equippedArmor == found.get()) equippedArmor = nullptr;
        return found;
    }

    void listInventory() const {
        if (inventory.empty()) { cout << "  (empty)" << endl; return; }
        for (const auto& item : inventory) cout << "  - " << item->describe() << endl;
    }

    void sortInventoryByValue() {
        sort(inventory.begin(), inventory.end(),
            [](const unique_ptr<Item>& a, const unique_ptr<Item>& b) {
                return a->getValue() > b->getValue();
            });
        cout << "Inventory sorted by value (highest first)." << endl;
    }

    int totalInventoryValue() const {
        return accumulate(inventory.begin(), inventory.end(), 0,
            [](int sum, const unique_ptr<Item>& i) { return sum + i->getValue(); });
    }

    int countPotions() const {
        return static_cast<int>(count_if(inventory.begin(), inventory.end(),
            [](const unique_ptr<Item>& i) { return dynamic_cast<Potion*>(i.get()) != nullptr; }));
    }

    void equipWeapon(Weapon* w) { equippedWeapon = w; cout << name << " equips " << w->getName() << "." << endl; }
    void equipArmor(Armor* a) { equippedArmor = a; cout << name << " equips " << a->getName() << "." << endl; }

    void heal(int amount) {
        health = min(maxHealth, health + amount);
        cout << name << " heals to " << health << "/" << maxHealth << " HP." << endl;
    }

    void takeDamage(int dmg) {
        int reduced = equippedArmor ? max(0, dmg - equippedArmor->getDefense()) : dmg;
        health -= reduced;
        if (health < 0) health = 0;
        cout << name << " takes " << reduced << " damage (" << health << "/" << maxHealth << " HP)." << endl;
    }

    int attackDamage() const { return equippedWeapon ? equippedWeapon->getDamage() : 5; }
    bool isAlive() const { return health > 0; }

    const string& getName() const { return name; }
    int getHealth() const { return health; }
    int getMaxHealth() const { return maxHealth; }
};

// Item::use() implementations — need the full Player definition above
void Potion::use(Player& player) {
    cout << player.getName() << " drinks the " << name << "." << endl;
    player.heal(healAmount);
}
void Weapon::use(Player& player) { player.equipWeapon(this); }
void Armor::use(Player& player) { player.equipArmor(this); }

// ============================================================
// ROOM — owns its Items and Enemies (unique_ptr); exits are
// weak_ptr observers into rooms owned elsewhere (World), which
// is what lets the room graph contain loops without leaking.
// ============================================================

class Room {
private:
    string name;
    string description;
    vector<unique_ptr<Item>> items;
    vector<unique_ptr<Enemy>> enemies;
    map<string, weak_ptr<Room>> exits;

public:
    Room(string n, string desc) : name(move(n)), description(move(desc)) {}

    void addItem(unique_ptr<Item> item) { items.push_back(move(item)); }
    void addEnemy(unique_ptr<Enemy> enemy) { enemies.push_back(move(enemy)); }

    void setExit(const string& direction, const shared_ptr<Room>& room) {
        exits[direction] = room; // stored as weak_ptr — does not extend room's lifetime
    }

    shared_ptr<Room> getExit(const string& direction) const {
        auto it = exits.find(direction);
        if (it == exits.end()) return nullptr;
        return it->second.lock(); // nullptr if the room somehow no longer exists
    }

    unique_ptr<Item> takeItem(const string& itemName) {
        auto it = find_if(items.begin(), items.end(),
            [&](const unique_ptr<Item>& i) { return i->getName() == itemName; });
        if (it == items.end()) return nullptr;
        unique_ptr<Item> found = move(*it);
        items.erase(it);
        return found;
    }

    bool hasLivingEnemies() const {
        return any_of(enemies.begin(), enemies.end(),
            [](const unique_ptr<Enemy>& e) { return e->isAlive(); });
    }

    Enemy* firstLivingEnemy() const {
        auto it = find_if(enemies.begin(), enemies.end(),
            [](const unique_ptr<Enemy>& e) { return e->isAlive(); });
        return it == enemies.end() ? nullptr : it->get();
    }

    void clearDeadEnemies() {
        enemies.erase(remove_if(enemies.begin(), enemies.end(),
            [](const unique_ptr<Enemy>& e) { return !e->isAlive(); }), enemies.end());
    }

    void describe() const {
        cout << "\n== " << name << " ==" << endl;
        cout << description << endl;
        if (!items.empty()) {
            cout << "Items here:" << endl;
            for (const auto& i : items) cout << "  - " << i->getName() << endl;
        }
        if (hasLivingEnemies()) {
            cout << "Enemies here:" << endl;
            for (const auto& e : enemies) if (e->isAlive()) cout << "  - " << e->describe() << endl;
        }
        if (!exits.empty()) {
            cout << "Exits:";
            for (const auto& [dir, room] : exits) cout << " " << dir;
            cout << endl;
        }
    }

    const string& getName() const { return name; }
};

// ============================================================
// WORLD — owns all rooms via shared_ptr (rooms' exits reference
// each other only weakly, so the loop in the map does not leak).
// ============================================================

class World {
private:
    vector<shared_ptr<Room>> rooms;
    shared_ptr<Room> currentRoom;
    Player player;
    set<string> visitedRooms;
    bool victory = false;

public:
    explicit World(string playerName) : player(move(playerName)) {}

    shared_ptr<Room> addRoom(const string& name, const string& desc) {
        auto room = make_shared<Room>(name, desc);
        rooms.push_back(room);
        return room;
    }

    void setCurrentRoom(const shared_ptr<Room>& room) {
        currentRoom = room;
        visitedRooms.insert(currentRoom->getName());
    }

    void look() const { currentRoom->describe(); }

    void go(const string& direction) {
        if (direction.empty()) { cout << "Go where?" << endl; return; }
        auto next = currentRoom->getExit(direction);
        if (!next) { cout << "You can't go that way." << endl; return; }
        currentRoom = next;
        visitedRooms.insert(currentRoom->getName());
        cout << "You head " << direction << "." << endl;
        currentRoom->describe();
    }

    void take(const string& itemName) {
        if (itemName.empty()) { cout << "Take what?" << endl; return; }
        auto item = currentRoom->takeItem(itemName);
        if (!item) { cout << "No such item here." << endl; return; }
        player.pickUp(move(item));
    }

    void useItem(const string& itemName) {
        if (itemName.empty()) { cout << "Use what?" << endl; return; }
        Item* item = player.findItem(itemName);
        if (!item) { cout << "You don't have that." << endl; return; }
        bool isPotion = dynamic_cast<Potion*>(item) != nullptr;
        item->use(player); // polymorphic: heals, or equips, depending on subclass
        if (isPotion) player.removeItem(itemName); // potions are single-use
    }

    void fight() {
        Enemy* enemy = currentRoom->firstLivingEnemy();
        if (!enemy) { cout << "Nothing here to fight." << endl; return; }
        cout << "You engage the " << enemy->getName() << "!" << endl;
        bool wasDragon = dynamic_cast<Dragon*>(enemy) != nullptr;

        while (enemy->isAlive() && player.isAlive()) {
            enemy->takeDamage(player.attackDamage());
            if (!enemy->isAlive()) break;
            player.takeDamage(enemy->attack());
        }

        if (player.isAlive()) {
            cout << "You defeated the " << enemy->getName() << "!" << endl;
            currentRoom->clearDeadEnemies();
            if (wasDragon) {
                victory = true;
                cout << "\n*** The dragon falls. You have conquered the dungeon! ***" << endl;
            }
        } else {
            cout << "You have been slain..." << endl;
        }
    }

    void status() const {
        cout << "\n-- Status --" << endl;
        cout << "Name: " << player.getName() << endl;
        cout << "HP: " << player.getHealth() << "/" << player.getMaxHealth() << endl;
        cout << "Rooms visited: " << visitedRooms.size() << endl;
        cout << "Inventory value: " << player.totalInventoryValue() << endl;
        cout << "Potions carried: " << player.countPotions() << endl;
    }

    bool hasWon() const { return victory; }
    int roomsVisited() const { return static_cast<int>(visitedRooms.size()); }
    Player& getPlayer() { return player; }
    Room* getCurrentRoom() const { return currentRoom.get(); }
};

// ============================================================
// MAIN — builds the world, then runs a simple text command loop
// ============================================================

void printHelp() {
    cout << "\nCommands:" << endl;
    cout << "  look                 - describe the current room" << endl;
    cout << "  go <direction>       - move (e.g. 'go north')" << endl;
    cout << "  take <item name>     - pick up an item from the room" << endl;
    cout << "  inventory            - list what you're carrying" << endl;
    cout << "  sort                 - sort inventory by value" << endl;
    cout << "  use <item name>      - drink a potion or equip a weapon/armor" << endl;
    cout << "  fight                - attack the first living enemy here" << endl;
    cout << "  status               - show HP, rooms visited, inventory value" << endl;
    cout << "  help                 - show this list" << endl;
    cout << "  quit                 - exit the game" << endl;
}

int main() {
    cout << "=== THE FORGOTTEN DUNGEON ===" << endl;
    cout << "Enter your character's name: ";
    string playerName;
    getline(cin, playerName);
    if (playerName.empty()) playerName = "Hero";

    World world(playerName);

    // Build the map. Note the loop: Entrance <-> Armory <-> Cave <-> Lair <-> Entrance.
    // This is a real cyclic graph — exactly the shape that would leak if exits
    // were shared_ptr instead of weak_ptr.
    auto entrance = world.addRoom("Entrance Hall", "A dusty stone hall. Torches flicker on the walls.");
    auto armory   = world.addRoom("Armory", "Racks of rusted weapons line the walls.");
    auto cave     = world.addRoom("Damp Cave", "Water drips somewhere in the darkness.");
    auto lair     = world.addRoom("Dragon's Lair", "The air is hot. Something huge is breathing nearby.");

    entrance->setExit("north", armory);
    entrance->setExit("east", lair);
    armory->setExit("south", entrance);
    armory->setExit("east", cave);
    cave->setExit("west", armory);
    cave->setExit("north", lair);
    lair->setExit("south", cave);
    lair->setExit("west", entrance);

    entrance->addItem(make_unique<Potion>("Minor Potion", 10, 20));
    armory->addItem(make_unique<Weapon>("Iron Sword", 50, 15));
    armory->addItem(make_unique<Armor>("Leather Armor", 40, 5));
    cave->addItem(make_unique<Potion>("Cave Potion", 15, 15));
    cave->addEnemy(make_unique<Goblin>("Cave Goblin"));
    lair->addItem(make_unique<Weapon>("Dragon Slayer", 200, 30));
    lair->addEnemy(make_unique<Dragon>("Ancient Dragon"));

    world.setCurrentRoom(entrance);
    world.look();
    cout << "\nType 'help' for a list of commands.\n";

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

        if (cmd == "quit" || cmd == "exit") {
            cout << "Farewell, " << world.getPlayer().getName() << "." << endl;
            break;
        } else if (cmd == "help") {
            printHelp();
        } else if (cmd == "look") {
            world.look();
        } else if (cmd == "go") {
            world.go(rest);
        } else if (cmd == "take") {
            world.take(rest);
        } else if (cmd == "inventory" || cmd == "i") {
            world.getPlayer().listInventory();
        } else if (cmd == "sort") {
            world.getPlayer().sortInventoryByValue();
        } else if (cmd == "use") {
            world.useItem(rest);
        } else if (cmd == "fight") {
            world.fight();
        } else if (cmd == "status") {
            world.status();
        } else if (cmd.empty()) {
            // ignore blank input
        } else {
            cout << "Unknown command. Type 'help'." << endl;
        }

        if (!world.getPlayer().isAlive()) {
            cout << "\nGAME OVER." << endl;
            break;
        }
        if (world.hasWon()) {
            cout << "\n-- Final Report --" << endl;
            cout << "Rooms explored: " << world.roomsVisited() << " / 4" << endl;
            cout << "Loot value carried: " << world.getPlayer().totalInventoryValue() << endl;
            break;
        }
    }

    return 0;
}
