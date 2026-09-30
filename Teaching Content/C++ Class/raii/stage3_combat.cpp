// stage3_combat.cpp
// Section 4: Enemy Hierarchy & Turn-Based Combat
// Builds on Stage 2 (items + Player) by adding Enemy/Goblin/Dragon and a fight loop.
// Compile: g++ -std=c++17 -o stage3_combat stage3_combat.cpp

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

using namespace std;

class Player; // forward declaration — Item::use() needs a reference to Player

// ---------------- Items (unchanged from Stage 2) ----------------

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

// ---------------- Enemies (new in this stage) ----------------

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

// ---------------- Player (extended with combat stats) ----------------

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

    void equipWeapon(Weapon* w) { equippedWeapon = w; cout << name << " equips " << w->getName() << "." << endl; }
    void equipArmor(Armor* a) { equippedArmor = a; cout << name << " equips " << a->getName() << "." << endl; }

    void heal(int amount) {
        health = min(maxHealth, health + amount);
        cout << name << " heals to " << health << "/" << maxHealth << " HP." << endl;
    }

    // --- new in this stage ---
    void takeDamage(int dmg) {
        int reduced = equippedArmor ? max(0, dmg - equippedArmor->getDefense()) : dmg;
        health -= reduced;
        if (health < 0) health = 0;
        cout << name << " takes " << reduced << " damage (" << health << "/" << maxHealth << " HP)." << endl;
    }

    int attackDamage() const { return equippedWeapon ? equippedWeapon->getDamage() : 5; } // bare fists = 5
    bool isAlive() const { return health > 0; }

    const string& getName() const { return name; }
    int getHealth() const { return health; }
};

void Potion::use(Player& player) {
    cout << player.getName() << " drinks the " << name << "." << endl;
    player.heal(healAmount);
}
void Weapon::use(Player& player) { player.equipWeapon(this); }
void Armor::use(Player& player) { player.equipArmor(this); }

// ---------------- Room (items + enemies) ----------------

class Room {
private:
    string name;
    vector<unique_ptr<Item>> items;
    vector<unique_ptr<Enemy>> enemies;
public:
    explicit Room(string n) : name(move(n)) {}

    void addItem(unique_ptr<Item> item) { items.push_back(move(item)); }
    void addEnemy(unique_ptr<Enemy> enemy) { enemies.push_back(move(enemy)); }

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
        if (!items.empty()) {
            cout << "Items here:" << endl;
            for (const auto& i : items) cout << "  - " << i->getName() << endl;
        }
        if (hasLivingEnemies()) {
            cout << "Enemies here:" << endl;
            for (const auto& e : enemies) if (e->isAlive()) cout << "  - " << e->describe() << endl;
        }
        if (items.empty() && !hasLivingEnemies()) cout << "There is nothing of interest here." << endl;
    }
};

// ---------------- Combat loop ----------------

void fight(Player& player, Room& room) {
    Enemy* enemy = room.firstLivingEnemy();
    if (!enemy) { cout << "Nothing here to fight." << endl; return; }

    cout << "You engage the " << enemy->getName() << "!" << endl;
    while (enemy->isAlive() && player.isAlive()) {
        enemy->takeDamage(player.attackDamage());
        if (!enemy->isAlive()) break;
        player.takeDamage(enemy->attack());
    }

    if (player.isAlive()) {
        cout << "You defeated the " << enemy->getName() << "!" << endl;
        room.clearDeadEnemies();
    } else {
        cout << "You have been slain..." << endl;
    }
}

// ---------------- Demo ----------------

int main() {
    cout << "=== STAGE 3: COMBAT DEMO ===" << endl;

    Room cave("Damp Cave");
    cave.addItem(make_unique<Potion>("Cave Potion", 15, 15));
    cave.addEnemy(make_unique<Goblin>("Cave Goblin"));

    Player player("Tester");
    player.pickUp(make_unique<Weapon>("Iron Sword", 50, 15));
    player.findItem("Iron Sword")->use(player); // equip it right away for the demo

    cave.describe();
    cout << "\nCommands: take <item>, use <item>, inventory, fight, quit\n";

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
        } else if (cmd == "take") {
            auto item = cave.takeItem(rest);
            if (!item) cout << "No such item here." << endl;
            else player.pickUp(move(item));
        } else if (cmd == "use") {
            Item* item = player.findItem(rest);
            if (!item) { cout << "You don't have that." << endl; continue; }
            bool isPotion = dynamic_cast<Potion*>(item) != nullptr;
            item->use(player);
            if (isPotion) player.removeItem(rest);
        } else if (cmd == "inventory") {
            player.listInventory();
        } else if (cmd == "fight") {
            fight(player, cave);
        } else {
            cout << "Unknown command." << endl;
        }

        if (!player.isAlive()) { cout << "GAME OVER." << endl; break; }
    }

    return 0;
}
