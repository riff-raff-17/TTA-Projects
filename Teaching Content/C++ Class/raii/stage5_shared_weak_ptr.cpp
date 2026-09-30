// stage5_shared_weak_ptr.cpp
// Section 7: The Room Graph Problem — shared_ptr, weak_ptr, and Cycles
// Builds on Stage 4 by bringing navigation back (from Stage 1) and upgrading
// Room::exits from a raw pointer to a weak_ptr, with a new World class that
// owns every Room via shared_ptr. This is the version that finally supports
// a room graph with real loops in it without leaking.
// Compile: g++ -std=c++17 -o stage5_shared_weak_ptr stage5_shared_weak_ptr.cpp

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

class Player;

// ---------------- Items (unchanged) ----------------

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
    void use(Player& player) override;
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
    void use(Player& player) override;
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
    void use(Player& player) override;
    int getDefense() const { return defense; }
    string describe() const override {
        ostringstream oss;
        oss << name << " (Armor, defense " << defense << ", worth " << value << ")";
        return oss.str();
    }
};

// ---------------- Enemies (unchanged) ----------------

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

// ---------------- Player (unchanged from Stage 4) ----------------

class Player {
private:
    string name;
    int health;
    int maxHealth;
    vector<unique_ptr<Item>> inventory;
    Weapon* equippedWeapon = nullptr;
    Armor* equippedArmor = nullptr;

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

void Potion::use(Player& player) {
    cout << player.getName() << " drinks the " << name << "." << endl;
    player.heal(healAmount);
}
void Weapon::use(Player& player) { player.equipWeapon(this); }
void Armor::use(Player& player) { player.equipArmor(this); }

// ---------------- Room — exits are now weak_ptr, not a raw pointer ----------------
// Rooms are owned by World via shared_ptr. Room::exits only *observes* other
// rooms via weak_ptr, which is what lets the exit map contain loops (Entrance
// -> Armory -> Entrance, etc.) without any room keeping another alive forever.

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

// ---------------- World — the one true owner of every Room ----------------

class World {
private:
    vector<shared_ptr<Room>> rooms;
    shared_ptr<Room> currentRoom;
    Player player;
    set<string> visitedRooms;

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
        item->use(player);
        if (isPotion) player.removeItem(itemName);
    }

    void fight() {
        Enemy* enemy = currentRoom->firstLivingEnemy();
        if (!enemy) { cout << "Nothing here to fight." << endl; return; }
        cout << "You engage the " << enemy->getName() << "!" << endl;
        while (enemy->isAlive() && player.isAlive()) {
            enemy->takeDamage(player.attackDamage());
            if (!enemy->isAlive()) break;
            player.takeDamage(enemy->attack());
        }
        if (player.isAlive()) {
            cout << "You defeated the " << enemy->getName() << "!" << endl;
            currentRoom->clearDeadEnemies();
        } else {
            cout << "You have been slain..." << endl;
        }
    }

    void status() const {
        cout << "HP: " << player.getHealth() << "/" << player.getMaxHealth() << endl;
        cout << "Rooms visited: " << visitedRooms.size() << endl;
        cout << "Inventory value: " << player.totalInventoryValue() << endl;
        cout << "Potions carried: " << player.countPotions() << endl;
    }

    Player& getPlayer() { return player; }
};

// ---------------- Demo ----------------

int main() {
    cout << "=== STAGE 5: SHARED_PTR / WEAK_PTR ROOM GRAPH DEMO ===" << endl;

    World world("Tester");

    // A genuine cycle: Entrance -> Armory -> Cave -> Entrance (via "west" shortcut).
    // If setExit() stored a shared_ptr instead of a weak_ptr, this loop alone
    // would be enough to leak all three rooms.
    auto entrance = world.addRoom("Entrance Hall", "A dusty stone hall.");
    auto armory   = world.addRoom("Armory", "Racks of rusted weapons line the walls.");
    auto cave     = world.addRoom("Damp Cave", "Water drips somewhere in the darkness.");

    entrance->setExit("north", armory);
    armory->setExit("south", entrance);
    armory->setExit("east", cave);
    cave->setExit("west", armory);
    cave->setExit("north", entrance); // the shortcut that completes the loop

    entrance->addItem(make_unique<Potion>("Minor Potion", 10, 20));
    armory->addItem(make_unique<Weapon>("Iron Sword", 50, 15));
    cave->addEnemy(make_unique<Goblin>("Cave Goblin"));

    world.setCurrentRoom(entrance);
    world.look();
    cout << "\nCommands: look, go <dir>, take <item>, use <item>, inventory, sort, status, fight, quit\n";

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

        if (cmd == "quit") break;
        else if (cmd == "look") world.look();
        else if (cmd == "go") world.go(rest);
        else if (cmd == "take") world.take(rest);
        else if (cmd == "use") world.useItem(rest);
        else if (cmd == "inventory") world.getPlayer().listInventory();
        else if (cmd == "sort") world.getPlayer().sortInventoryByValue();
        else if (cmd == "status") world.status();
        else if (cmd == "fight") world.fight();
        else cout << "Unknown command." << endl;

        if (!world.getPlayer().isAlive()) { cout << "GAME OVER." << endl; break; }
    }

    return 0;
}
