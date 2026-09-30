# Capstone Project: The Forgotten Dungeon — 2-Hour Extended Build

Companion file: `dungeon_crawler.cpp` (the complete, compiling target you'll build toward).
This session is the direct sequel to the RAII/Smart Pointers/STL lesson — it assumes that
lesson (or equivalent comfort with `unique_ptr`, `shared_ptr`, `weak_ptr`, move semantics,
and basic STL containers/algorithms) and turns those separate ideas into one working program.

Where the original lesson ended with a small starter skeleton (`Item`/`Room`/`Player`,
one item moving from one room to one player), this session grows that into a full,
playable, four-room text adventure with combat, equipment, and a win condition —
and along the way, hits a smart-pointer problem the starter skeleton was too small to expose:
**a room graph has loops, and loops are exactly where naive `shared_ptr` ownership leaks.**

---

## 0. Setup (before class starts)
- Have the original starter skeleton compiled and ready to extend, or start fresh from this file's Section 1.
- Confirm the compiler supports C++17 (`-std=c++17`) — structured bindings and `if (auto x = ...)` are used throughout.
- Have `dungeon_crawler.cpp` open in a second window as the "answer key" to reveal incrementally, not to paste in wholesale.
- Two hours is enough to *build* the whole thing live, but not to polish it — set expectations that this is a working prototype, not production code.

---

## 1. Recap & Architecture Planning (10 min)

**Talking points:**
- Quick recap of the starter skeleton: one `Item`, one `Room`, one `Player`, one `move`. That proved ownership transfer works. Today we scale it up in every direction at once.
- Before writing code, whiteboard the full ownership map — this is the most important five minutes of the session, because every subsequent decision follows from it:
  - `Item`s — owned by exactly one `vector<unique_ptr<Item>>` at a time (a `Room`'s or the `Player`'s). Never shared.
  - `Enemy`s — owned by the `Room` they're in, via `vector<unique_ptr<Enemy>>`.
  - `Room`s — owned by a `World` class, via `vector<shared_ptr<Room>>`. (We'll get to *why shared_ptr* in Section 7 — for now just flag that rooms are different from items/enemies because rooms need to reference each other.)
  - The equipped weapon/armor a `Player` is holding — not owned twice; it's a **raw, non-owning pointer** back into the inventory `vector` that already owns it. This is the payoff of understanding ownership: you can now use a raw pointer safely, because you know exactly who owns the object and that its lifetime outlives the pointer's use.

**Key line to say:**
> "Every class we write today, ask the same question first: who owns this, and is there ever a moment where two owners could both think they're responsible for deleting it? If the answer to the second half is no, you've picked the right smart pointer — or decided a raw pointer is fine after all."

**Ask the audience:** "Items and enemies live in exactly one room. Rooms... don't live in exactly one place, do they? A room has neighbors in every direction, and some of those neighbors point back. What does that do to a naive `shared_ptr` design?" — plant the question, don't answer it yet. Section 7 pays it off.

---

## 2. Building the World: Multi-Room Map & Navigation (15 min)

**Talking points:**
- We need more than one `Room`, and rooms need to know about their neighbors — that's a `map<string, ???>` from direction name to a room reference. What goes in the `???` is deliberately left open for now (raw pointer? `shared_ptr`? `weak_ptr`?) — we'll build it with a raw pointer first to get navigation working, then upgrade it in Section 7 once the cycle problem is concrete.
- Introduce a `World` class as the owner of all rooms — this is new compared to the starter skeleton, where `main()` owned everything directly. With four-plus rooms and a graph of exits between them, ownership needs a home that isn't a local variable.

**Live-code — expand `Room`:**
```cpp
class Room {
private:
    string name;
    string description;
    vector<unique_ptr<Item>> items;
    vector<unique_ptr<Enemy>> enemies;   // Enemy doesn't exist yet — stub it or skip to Section 4 first
    map<string, Room*> exits;            // raw pointer for now — deliberately not the final version

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
```

**Live-code — wire up a small map in `main()` and navigate it:**
```cpp
vector<unique_ptr<Room>> rooms;
rooms.push_back(make_unique<Room>("Entrance Hall", "A dusty stone hall."));
rooms.push_back(make_unique<Room>("Armory", "Racks of rusted weapons line the walls."));

rooms[0]->setExit("north", rooms[1].get());
rooms[1]->setExit("south", rooms[0].get());

Room* current = rooms[0].get();
current->describe();
current = current->getExit("north");
current->describe();
```

**Key line to say:**
> "Notice `rooms[1].get()` — we're handing out a raw, non-owning pointer to something a `vector<unique_ptr<Room>>` still owns. That's fine *as long as the vector outlives the pointer's use*, which it does here. Keep that condition in your head; Section 7 is about a case where it stops being true."

**Quick practice (work in pairs, 3 min):** Add a third room and wire up its exits so all three connect in a line (A ↔ B ↔ C). Navigate from A to C and back.

---

## 3. Item Hierarchy: Weapon, Armor, Potion (15 min)

**Talking points:**
- The starter skeleton had `Item` → `Potion` only. Real inventory systems need items that *do different things* when used — heal you, equip as a weapon, equip as armor. That's `virtual void use(Player&)`, overridden per subclass.
- This is also the first time `Item::use()` needs to reach back into `Player` — a forward declaration (`class Player;`) up top lets `Item` declare the method; the actual body goes below `Player`'s full definition.

**Live-code:**
```cpp
class Player; // forward declaration

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

class Weapon : public Item {
    int damage;
public:
    Weapon(string n, int v, int dmg) : Item(move(n), v), damage(dmg) {}
    void use(Player& player) override; // defined after Player exists
    int getDamage() const { return damage; }
    string describe() const override { /* ... */ }
};
```
- Repeat the pattern for `Armor` (a `defense` stat) and `Potion` (a `healAmount` stat, carried over from the starter skeleton).

**Key line to say:**
> "`use()` being pure virtual means the compiler is doing our dispatch for us — `item->use(player)` runs the right behavior whether `item` points at a `Weapon`, `Armor`, or `Potion`, and we never write an if/else chain checking types. That's the whole point of polymorphism showing up as a design decision, not just a vocabulary word."

**Quick practice (5 min):** Sketch (on paper or in comments) what `Weapon::use()` and `Armor::use()` should do differently from `Potion::use()`. (Answer to reveal: potions consume themselves and heal; weapons/armor don't get consumed, they just update a pointer on `Player` saying "this is equipped now.")

---

## 4. Enemy Hierarchy & Turn-Based Combat (20 min)

**Talking points:**
- Same polymorphism pattern, new hierarchy: `Enemy` base class, `Goblin` and `Dragon` subclasses with different stats and (for the dragon) different `attack()` behavior.
- Combat itself is a simple turn loop — nothing new conceptually, but it's the first time our smart-pointer-owned objects do something *stateful and repeated* over multiple turns, which is a good test that ownership is solid (nobody's getting double-deleted mid-fight).

**Live-code:**
```cpp
class Enemy {
protected:
    string name;
    int health;
    int attackPower;
public:
    Enemy(string n, int hp, int atk) : name(move(n)), health(hp), attackPower(atk) {}
    virtual ~Enemy() { cout << "  [" << name << " is gone.]" << endl; }
    bool isAlive() const { return health > 0; }
    void takeDamage(int dmg) { health = max(0, health - dmg); }
    virtual int attack() const { return attackPower; }
    const string& getName() const { return name; }
};

class Dragon : public Enemy {
public:
    explicit Dragon(string n = "Dragon") : Enemy(move(n), 60, 12) {}
    int attack() const override {
        cout << "  " << name << " breathes fire!" << endl;
        return attackPower; // could scale this up — dragons hit harder than the base implies
    }
};
```

**Live-code — the fight loop:**
```cpp
void fight(Player& player, Enemy* enemy) {
    while (enemy->isAlive() && player.isAlive()) {
        enemy->takeDamage(player.attackDamage());
        if (!enemy->isAlive()) break;
        player.takeDamage(enemy->attack());
    }
}
```

**Key line to say:**
> "`player.attackDamage()` should read from whatever weapon is equipped — this is where Section 1's raw non-owning pointer earns its keep. `Player` doesn't own a copy of the weapon, it just asks the one true owner, the inventory vector, through a pointer, what damage it does."

**Room integration:** give `Room` a `vector<unique_ptr<Enemy>>`, plus `firstLivingEnemy()` (using `find_if`) and `clearDeadEnemies()` (using the erase-remove idiom with `remove_if`) — both previewed here, expanded on in Section 6.

---

### ☕ Break (5 min)

---

## 5. (folded into Section 6 below — see note)

*Note: in the original 2-hour lesson this was Move Semantics as its own block. In this session, move semantics doesn't get a separate slot — it's already threaded through every `pickUp`, `takeItem`, and `push_back(move(item))` call from Sections 2–4. Use this transition to say so explicitly:*

**Key line to say:**
> "Notice we haven't had a 'move semantics' section today — and that's the point. Once it's internalized, `move` stops being a topic and starts being a reflex every time a `unique_ptr` changes hands. If you catch yourself writing a `unique_ptr` copy anywhere today, that compile error is `move` reminding you it's still there."

---

## 6. Inventory Management with STL Algorithms (15 min)

**Talking points:**
- `Player`'s inventory is a `vector<unique_ptr<Item>>` — now that it can hold real quantities of different item types, this is where `<algorithm>` and `<numeric>` earn their place instead of being toy examples.
- Four algorithms, each solving a real need in the game:
  - `sort` — order inventory by value, for display.
  - `find_if` — locate an item by name when the player types `use <item>`.
  - `count_if` — how many potions is the player carrying (uses `dynamic_cast` in the predicate to check subtype).
  - `accumulate` — total inventory value, for the end-of-game report.

**Live-code:**
```cpp
void sortInventoryByValue() {
    sort(inventory.begin(), inventory.end(),
        [](const unique_ptr<Item>& a, const unique_ptr<Item>& b) {
            return a->getValue() > b->getValue();
        });
}

int totalInventoryValue() const {
    return accumulate(inventory.begin(), inventory.end(), 0,
        [](int sum, const unique_ptr<Item>& i) { return sum + i->getValue(); });
}

int countPotions() const {
    return count_if(inventory.begin(), inventory.end(),
        [](const unique_ptr<Item>& i) { return dynamic_cast<Potion*>(i.get()) != nullptr; });
}
```

**Key line to say:**
> "Every one of these lambdas takes `const unique_ptr<Item>&` — a reference, never a copy. That's not a style choice, it's a requirement: `unique_ptr` can't be copied, so if you'd written `unique_ptr<Item>` by value in that lambda signature, it wouldn't compile. The algorithm's job is to iterate; ownership never moves just because you're looking at something."

**Also cover, briefly:** `Room::firstLivingEnemy()` (`find_if`), `Room::hasLivingEnemies()` (`any_of`), and `Room::clearDeadEnemies()` (`remove_if` + `erase` — the classic erase-remove idiom, worth naming explicitly since it looks unfamiliar the first time).

**Quick practice (5 min):** Using `count_if`, write a one-liner that checks whether the player is carrying at least one `Weapon` before letting them enter the Dragon's Lair.

---

## 7. The Room Graph Problem: `shared_ptr`, `weak_ptr`, and Cycles (15 min)

**Talking points:**
- Time to pay off Section 1's planted question. Go back to the raw-pointer `Room::exits` from Section 2 and build out the *full* map: Entrance ↔ Armory ↔ Cave ↔ Dragon's Lair, **and** a shortcut back from the Lair to the Entrance. Draw it on the board — this is a graph with a cycle in it, same shape as any real map with more than one way to loop back to where you started.
- With raw pointers, this was already fine, *because* `World` (or `main`) held every `Room` alive in a `vector<unique_ptr<Room>>` for the program's whole run, and the raw pointers in `exits` never outlived that. But now ask: what if two separate parts of the program each wanted partial ownership of rooms — say, a `World` and a `Quest` system that both need to keep a room alive? That's when people reach for `shared_ptr` on *both* sides, and that's exactly where it goes wrong.
- **The problem, concretely:** if `Room::exits` were `map<string, shared_ptr<Room>>`, then Entrance holds a `shared_ptr` to Armory, Armory holds one back to Entrance — a reference cycle. Even after `World`'s own `vector<shared_ptr<Room>>` is destroyed, Entrance and Armory still hold `shared_ptr`s to each other, so neither one's reference count ever reaches zero. Both leak, forever, silently — no crash, no leaked-memory warning at the point of the bug, just objects that never get cleaned up.
- **The fix:** `World` owns rooms via `shared_ptr` (the "real" ownership). `Room::exits` becomes `map<string, weak_ptr<Room>>` — an *observing* reference that doesn't add to the reference count. To actually use one, call `.lock()`, which hands back a `shared_ptr` if the room still exists, or an empty one if it's somehow gone.

**Live-code — upgrade `Room::exits` and `World`:**
```cpp
class Room {
    // ...
    map<string, weak_ptr<Room>> exits;
public:
    void setExit(const string& direction, const shared_ptr<Room>& room) {
        exits[direction] = room; // stored as weak_ptr — does not extend room's lifetime
    }
    shared_ptr<Room> getExit(const string& direction) const {
        auto it = exits.find(direction);
        if (it == exits.end()) return nullptr;
        return it->second.lock();
    }
};

class World {
    vector<shared_ptr<Room>> rooms; // the one true owner of every room
public:
    shared_ptr<Room> addRoom(const string& name, const string& desc) {
        auto room = make_shared<Room>(name, desc);
        rooms.push_back(room);
        return room;
    }
};
```

**Key line to say:**
> "This is the same fix as the `Door`/`Room` example from the smart-pointers lesson, but now it's not a toy — a dungeon map is *always* going to have loops in it, so this isn't a rare edge case you might hit, it's a design decision you make on day one for any graph-shaped ownership problem: one owner holds `shared_ptr`s, everyone else that just needs to *reference* it holds `weak_ptr` and calls `.lock()` when they actually need to use it."

**Quick practice (5 min, discussion not code):** "Where else in this game might you be tempted to reach for `shared_ptr` in both directions? What about a `Quest` that references a `Room`, and a `Room` that lists which quests are tied to it?" (No need to build it today — just get them naming the pattern.)

---

## 8. Full Game Loop Integration (20 min)

**Talking points:**
- All the pieces exist now: multi-room navigation, items with different `use()` behaviors, enemies with combat, inventory algorithms, and a leak-safe room graph. This section wires them into one playable `main()` with a text command loop.
- Keep the command parser simple on purpose — this isn't the lesson. One line in, split on the first space into a command word and the rest of the line as an argument (item and direction names can contain spaces, like "Iron Sword", so don't split on every space).

**Live-code — the command loop skeleton:**
```cpp
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
    else if (cmd == "fight") world.fight();
    else if (cmd == "inventory") world.getPlayer().listInventory();
    else if (cmd == "status") world.status();
    else cout << "Unknown command." << endl;

    if (!world.getPlayer().isAlive()) { cout << "GAME OVER." << endl; break; }
    if (world.hasWon()) { /* print final report using accumulate/set from Section 6 */ break; }
}
```

**Win condition:** defeating the `Dragon` in the Dragon's Lair sets a `victory` flag on `World`; the loop checks it after every `fight` command and prints a final report — rooms visited (`set<string>::size()`), total loot value (`accumulate`), using exactly the STL tools built in Section 6.

**Key line to say:**
> "Walk through what happens the instant the player types `use Iron Sword`: `findItem` (a `find_if` under the hood) locates it in the inventory vector by reference — no copy, no move, just a raw observing pointer, because we're only looking, not transferring ownership. Then `item->use(player)` dispatches polymorphically to `Weapon::use`, which sets `Player`'s non-owning `equippedWeapon` pointer. Every concept from today just fired in about three lines, and none of them stepped on each other."

**Live-play (remaining time):** run the finished program together, narrate what's happening at each `unique_ptr`/`shared_ptr` boundary as the group plays through to the dragon fight.

---

## 9. Wrap-up, Playtest, Stretch Challenges (5 min)

**Recap out loud, in order:**
1. **Ownership map first, code second** — every class's smart-pointer choice followed from an explicit answer to "who owns this, and could two owners exist at once?"
2. **`unique_ptr`** for items and enemies — single, unambiguous owner, moved on transfer.
3. **Raw pointers, used deliberately** — equipped weapon/armor, because the inventory vector's lifetime is known to outlive their use.
4. **`shared_ptr` + `weak_ptr`** for the room graph — the real reason `weak_ptr` exists: breaking cycles in ownership graphs that have loops by design, not by accident.
5. **STL algorithms** doing real work — `sort`, `find_if`, `count_if`, `accumulate`, `any_of`, `remove_if` — each solving an actual gameplay need, not a contrived example.

**Leave them with a challenge:**
> "Extend the game with a fifth room and a locked door that needs a `Key` item to open — `go` should refuse to move through a locked exit unless the key is in the player's inventory. Then add a second boss enemy in that new room with its own `attack()` override, the same way `Dragon` overrode it. If you're feeling ambitious: add a `Quest` class that a `Room` can reference — and decide for yourself, using Section 7's reasoning, whether that reference should be a `shared_ptr` or a `weak_ptr`, and why."

---

## Timing cheat sheet

| Section                                            | Minutes | Running total |
|-----------------------------------------------------|---------|----------------|
| Recap & Architecture Planning                        | 10      | 10             |
| Building the World: Multi-Room Map & Navigation      | 15      | 25             |
| Item Hierarchy: Weapon, Armor, Potion                | 15      | 40             |
| Enemy Hierarchy & Turn-Based Combat                  | 20      | 60             |
| Break                                                | 5       | 65             |
| Inventory Management with STL Algorithms             | 15      | 80             |
| The Room Graph Problem: shared_ptr, weak_ptr, Cycles | 15      | 95             |
| Full Game Loop Integration                           | 20      | 115            |
| Wrap-up, Playtest, Stretch Challenges                | 5       | 120            |

**Running long?** Cut points, in order of safety:
1. In Section 2, skip building a 3-room practice map live — go straight to the 4-room final map in Section 7's code.
2. In Section 4, cover `Goblin` only live-code; describe `Dragon`'s `attack()` override verbally and reveal the code from the answer key.
3. Shorten Section 8 to reading through the finished command loop together rather than live-typing it; spend the saved time on live-play instead.

**Running short / audience is fast?**
- Have them predict, before running it, what happens if `Room::exits` is left as `shared_ptr` and you try to destroy the `World` — ask them to reason through the leak before Section 7 confirms it.
- Ask them to add a `status` command enhancement: use `transform` to build a formatted list of "item: value" strings for display, instead of the plain loop in `listInventory()`.
- Let pairs race to add a fifth item type (e.g. `Scroll`, single-use like `Potion` but with a different effect) end-to-end: class, `use()` override, placed in a room, picked up and used in a live playtest.
