# RAII, Smart Pointers & STL in C++ — 2-Hour Lesson: Talking Points

Companion file: `resources.cpp` (live-code this alongside the talking points below).
This lesson is self-contained — every class used (`Engine`, `Car`, and later `Item`/`Room`/`Player`) is written from scratch below, so no prior code file is required. It pairs naturally with an earlier lesson on C++ OOP basics (classes, encapsulation, constructors/destructors, inheritance) if you've done one, but nothing here assumes you have.

---

## 0. Setup (before class starts)
- Have a blank `.cpp` file ready, or `resources.cpp` open with later stages commented out.
- Compiler/IDE ready to run code instantly — make sure it supports C++11 or later (smart pointers and move semantics need it; `<memory>`, `<algorithm>`, `<numeric>`).
- Basic comfort with C++ classes (writing a class with member variables, methods, and a constructor) is assumed — if that's brand new, a short primer beforehand will help, but the classes used in this lesson are simple and built up from nothing on the board.
- With two hours, plan for a real 5-minute break around the halfway mark (after Smart Pointers) — this is a lot of new mental model to hold at once.


---

## 1. Refresher: Pointers, `new`, and Heap vs. Stack (10 min)

**Talking points:**
- Everything after this depends on being comfortable with raw pointers first — smart pointers are just a wrapper around this idea, so the syntax needs to feel familiar before we build on it.
- **Stack vs. heap:** ordinary variables (`int x = 5;`, `Engine e;`) live on the **stack** and are cleaned up automatically when they go out of scope. Memory created with **`new`** lives on the **heap**, which is *not* cleaned up automatically — it stays allocated until something explicitly frees it.
- A **pointer** is a variable that stores a memory *address* rather than a value directly. `Engine*` means "a variable that holds the address of an `Engine`," not an `Engine` itself.
- **`new`** does two things: allocates space on the heap, and runs the constructor there. It hands back the address of what it just built.
- **`->`** means "follow this pointer to the object it points to, then access a member on it." It's shorthand for `(*e).start()` — dereference (`*e`, "the object at this address") then access with `.`.

**Live demo:**
```cpp
Engine* e = new Engine();  // "Engine built." prints; e now holds an address
e->start();                // follow the address, call start(); "Vroom." prints
```
- Draw it out on the board as two boxes: a small stack box named `e` containing an address, with an arrow pointing to a bigger heap box holding the actual `Engine`.

**Ask the audience:** "If `e` goes out of scope right now, what happens to the `Engine` it points to?" — let them guess before you show them the leak demo in the next section.

**Key line to say:**
> "A pointer is just an address — a sticky note, not the house. Destroying the sticky note doesn't destroy the house. That gap between 'the pointer is gone' and 'the object is gone' is exactly what causes memory leaks, and exactly what RAII is about to fix."

---

## 2. Why RAII? (10 min)

**Talking points:**
- Last time we saw a destructor run automatically — that wasn't just a cute feature, it's the foundation of how C++ manages *any* resource (memory, files, network sockets, locks).
- **RAII = Resource Acquisition Is Initialization.** The idea: tie a resource's lifetime to an object's lifetime. Acquire the resource in the constructor, release it in the destructor. No manual cleanup calls to remember.
- Contrast with the manual way, which is where bugs live:
  ```cpp
  Engine* engine = new Engine();
  // ... do stuff ...
  // forgot to call delete engine; -> memory leak
  // or an exception is thrown before delete -> leak anyway
  ```
- **Say something like:**
  > "Every `new` you write is a promise to write a matching `delete` — on every code path, including ones where an exception jumps out early. RAII means you stop making that promise manually and let an object's lifetime make it for you."
- It's not just memory. The exact same pattern protects file handles, network sockets, mutex locks — anything with an "open/close" or "acquire/release" shape. Once you see RAII in one context, you'll recognize it everywhere in C++ (and in libraries that mimic it, like `std::lock_guard` for mutexes).

**Live demo — the problem:**
```cpp
class Engine {
public:
    Engine() { cout << "Engine built." << endl; }
    ~Engine() { cout << "Engine destroyed." << endl; }
    void start() { cout << "Vroom." << endl; }
};

void leaky() {
    Engine* e = new Engine();
    e->start();
    // no delete e; -> "Engine destroyed." never prints
}
```
- Run it, point out the missing destructor message. That's a leaked `Engine`.

**Live demo — the same bug, but sneakier (exceptions):**
```cpp
void leakyWithException(bool badInput) {
    Engine* e = new Engine();
    if (badInput) {
        throw std::runtime_error("bad input");  // jumps out immediately —
        // delete e; below never runs, even if you remembered to write it!
    }
    e->start();
    delete e;
}
```
**Key line to say:**
> "This is the part that makes manual memory management genuinely hard, not just tedious. You can write `delete` on every 'normal' path and still leak, because exceptions skip past your cleanup code entirely. RAII closes this hole automatically, because destructors run during exception unwinding too — that's a guarantee the language makes."

---

## 3. Smart Pointers (25 min)

**Talking points:**
- A **smart pointer** is a small wrapper class around a raw pointer that follows RAII: it deletes what it owns automatically when it goes out of scope — including during exception unwinding, which is exactly the gap we just found.
- Three you need to know, in order of how often you'll use them:
  - **`std::unique_ptr`** — exactly one owner. Cannot be copied, only *moved*. This should be your default choice.
  - **`std::shared_ptr`** — multiple owners, reference-counted. The object is destroyed when the last owner goes away.
  - **`std::weak_ptr`** — a non-owning observer of a `shared_ptr`. Used to break reference cycles.

**Live demo — `unique_ptr`:**
```cpp
#include <memory>

void notLeakyAnymore() {
    std::unique_ptr<Engine> e = std::make_unique<Engine>();
    e->start();
    // no delete needed — destructor runs automatically here
}
```
- Run it — show "Engine destroyed." prints without writing `delete` anywhere.
- Try to compile `std::unique_ptr<Engine> e2 = e;` — show the compile error. **Key line to say:**
  > "That error is the language protecting you. Two `unique_ptr`s can't both think they own the same object — someone would double-delete it."
- Show the fix: `std::unique_ptr<Engine> e2 = std::move(e);` — ownership transfers, `e` is now empty. (We'll dig into exactly what `std::move` does in the next section.)
- Now re-run the exception demo from before, but with `unique_ptr` instead of a raw pointer:
  ```cpp
  void safeWithException(bool badInput) {
      std::unique_ptr<Engine> e = std::make_unique<Engine>();
      if (badInput) {
          throw std::runtime_error("bad input");
          // no delete needed — e's destructor runs during unwinding anyway
      }
      e->start();
  }
  ```
  Run it with `badInput = true` and point out `"Engine destroyed."` still prints, even though we threw before reaching the end of the function.

**Live demo — `shared_ptr`:**
```cpp
#include <memory>

void sharedDemo() {
    std::shared_ptr<Engine> e1 = std::make_shared<Engine>();
    cout << "Owners: " << e1.use_count() << endl; // 1
    {
        std::shared_ptr<Engine> e2 = e1; // copy is allowed
        cout << "Owners: " << e1.use_count() << endl; // 2
    } // e2 destroyed, but Engine survives — e1 still owns it
    cout << "Owners: " << e1.use_count() << endl; // 1
} // now the Engine is destroyed
```
- Run it and narrate the `use_count()` changes — this makes the reference counting visible instead of abstract.

**Live demo — `weak_ptr` (the reference cycle problem):**
- Set up the problem first: two objects that both hold a `shared_ptr` to each other will never reach a reference count of zero, even when nothing outside points to either of them. That's a leak `shared_ptr` alone can't fix.
  ```cpp
  struct Room; // forward declare

  struct Door {
      std::shared_ptr<Room> connectsTo;
      ~Door() { cout << "Door destroyed." << endl; }
  };

  struct Room {
      std::shared_ptr<Door> door;
      ~Room() { cout << "Room destroyed." << endl; }
  };

  void cycleLeak() {
      auto room = std::make_shared<Room>();
      auto door = std::make_shared<Door>();
      room->door = door;
      door->connectsTo = room;
      // neither destructor prints when this function ends —
      // room and door each keep the other's count above zero forever
  }
  ```
- Fix: make one side of the relationship a `std::weak_ptr` instead — it observes without adding to the count.
  ```cpp
  struct Door {
      std::weak_ptr<Room> connectsTo; // no longer keeps Room alive
      ~Door() { cout << "Door destroyed." << endl; }
  };
  ```
  To actually use a `weak_ptr`, you call `.lock()`, which returns a `shared_ptr` if the object is still alive, or an empty one if it's already been destroyed:
  ```cpp
  if (auto room = door->connectsTo.lock()) {
      // safe to use `room` here
  } else {
      cout << "Room is gone." << endl;
  }
  ```
**Key line to say:**
> "You won't reach for `weak_ptr` often, but when you have two objects that need to know about each other — a parent and child, a room and a door back to it — this is the tool that stops them from keeping each other alive forever."

**Tie it back to the Car:**
```cpp
class Car {
private:
    std::unique_ptr<Engine> engine;
    int speed = 0;
public:
    Car() : engine(std::make_unique<Engine>()) {
        cout << "Car built with its own engine." << endl;
    }
    void start() { engine->start(); }
    void accelerate() { speed += 10; }
    int getSpeed() const { return speed; }
};
```
**Say something like:**
> "The `Car` owns its `Engine` — not by name only, but literally: when the `Car` is destroyed, its `unique_ptr` destroys the `Engine` too, automatically, in the right order. No destructor code needed on our part."

**Rule of thumb to leave them with:**
> "Reach for `unique_ptr` by default. Reach for `shared_ptr` only when you genuinely need multiple owners. Reach for `weak_ptr` only to break a cycle between `shared_ptr`s. Avoid raw `new`/`delete` in your own code from here on."

**Quick practice (5 min, work in pairs):**
> "Rewrite `leakyWithException` from Section 2, but this time using `std::shared_ptr` instead of `unique_ptr`. Does anything actually behave differently for this single-owner case? (Answer to reveal after: no — this is exactly why `unique_ptr` should be your default. `shared_ptr` has reference-counting overhead you don't need when there's only ever one owner.)"

---

### ☕ Break (5 min)

---

## 4. Move Semantics (15 min)

**Talking points:**
- We used `std::move` a few minutes ago without really explaining it. Time to open that up, because it's central to how `unique_ptr` — and the STL containers we're about to cover — work efficiently.
- A **copy** duplicates data. A **move** transfers ownership of existing data to a new owner and leaves the original empty. Moving is cheap (just reassigning a few pointers); copying can be expensive (duplicating everything).
- `std::move` doesn't actually move anything itself — it just casts its argument to something called an "rvalue reference," which tells the compiler "this object's contents are up for grabs, you don't need to preserve it." The real work happens in a **move constructor**.
- This is exactly why `unique_ptr` can't be copied but can be moved: copying would mean two owners of the same heap object (unsafe), but moving means ownership cleanly transfers to the new variable while the old one becomes empty (safe).

**Live demo:**
```cpp
std::unique_ptr<Engine> a = std::make_unique<Engine>();
std::unique_ptr<Engine> b = std::move(a);

if (!a) {
    cout << "a is now empty." << endl;
}
b->start(); // b owns the Engine now
```
- Run it, point out `a` is empty (`nullptr`) after the move — nothing was duplicated, ownership just changed hands.

**Why this matters beyond smart pointers:**
- `std::vector` uses move semantics internally. When a `vector` resizes and needs to relocate its elements, it *moves* them into the new memory block rather than copying them, if the element type supports moving (which `unique_ptr` does, and raw structs/classes do by default).
- This is also why you'll see functions take `std::vector<std::unique_ptr<Item>>&&` or use `std::move(item)` when putting a `unique_ptr` into a container — you're explicitly handing ownership over, not asking for a (illegal) copy.

**Live demo, ties directly into the project:**
```cpp
std::vector<std::unique_ptr<Engine>> engines;
std::unique_ptr<Engine> e = std::make_unique<Engine>();
engines.push_back(std::move(e)); // must move — can't copy a unique_ptr
// e is now empty; the vector owns the Engine
```
**Key line to say:**
> "Any time you have a `unique_ptr` and need to hand it to a container, a function, or another object, you'll write `std::move(...)`. Seeing that pattern should now read as 'ownership is transferring here,' not just syntax you have to remember."

---

## 5. The Standard Template Library — Containers (20 min)

**Talking points:**
- The STL is a library of ready-made containers and algorithms. You've been hand-writing things (like fixed-size collections) that the STL already solved.
- **`std::vector<T>`** — a resizable array. Your default container for "a list of things," contiguous in memory, fast random access.
- **`std::map<K, V>`** — a sorted key/value lookup table. Good when you need keys in order, or don't know in advance how many entries you'll have.
- **`std::unordered_map<K, V>`** — same idea as `map`, but unordered and typically faster for lookups (hash table instead of a sorted tree). Use this when you don't care about order and just want speed.
- **`std::set<T>`** — like a `map` but with no separate value, just unique keys in sorted order. Good for "have I seen this before?" checks.
- Both containers happily hold smart pointers, which is how RAII and STL work together: a `vector` of `unique_ptr<Engine>` owns *all* those engines, and destroying the vector destroys every one of them.

**Live demo — `vector` of owning pointers:**
```cpp
#include <vector>
#include <map>
#include <string>

void garage() {
    std::vector<std::unique_ptr<Car>> cars;
    cars.push_back(std::make_unique<Car>());
    cars.push_back(std::make_unique<Car>());

    for (const auto& car : cars) {
        car->start();
    }
    // when `cars` goes out of scope, every Car AND every Engine
    // inside it is destroyed automatically. No leaks, no manual cleanup.
}
```
- Run it, count the destructor messages out loud — two `Engine`s, two `Car`s, all cleaned up with zero `delete` statements.

**Quick `map` example:**
```cpp
std::map<std::string, int> speedByColor;
speedByColor["red"] = 120;
speedByColor["blue"] = 95;
cout << "Red car speed: " << speedByColor["red"] << endl;

// iterate in sorted key order:
for (const auto& [color, speed] : speedByColor) {
    cout << color << ": " << speed << endl;
}
```
- Point out the structured binding `const auto& [color, speed]` — a clean C++17 way to unpack a `pair` from a `map` without `.first`/`.second`.

**Quick `unordered_map` and `set` examples:**
```cpp
#include <unordered_map>
#include <set>

std::unordered_map<std::string, int> fastLookup;
fastLookup["red"] = 120; // same interface as map, different internals

std::set<std::string> visitedRooms;
visitedRooms.insert("Entrance");
visitedRooms.insert("Entrance"); // no-op, already present
cout << "Rooms visited: " << visitedRooms.size() << endl; // 1
```
**Key line to say:**
> "`map` vs `unordered_map` is a decision you'll make often: do you need sorted order or iteration reproducibility? Use `map`. Do you just need fast lookups and don't care about order? Use `unordered_map`. When in doubt, `map` is the safer, more predictable default — optimize to `unordered_map` later if profiling says you need it."

---

## 6. STL Algorithms (15 min)

**Talking points:**

- The STL isn't just containers — `<algorithm>` (and `<numeric>`) gives you common operations so you stop writing manual loops for everything.
- Core ones to know: `std::sort`, `std::find_if`, `std::count_if`, `std::transform`, `std::accumulate`.
- The common shape: most algorithms take a **begin/end iterator pair** describing a range, and often a **lambda** describing what to do at each element.

**Live demo — `sort` and `find_if`:**

```cpp
#include <algorithm>

std::vector<int> speeds = {50, 120, 30, 95};
std::sort(speeds.begin(), speeds.end());
// speeds is now {30, 50, 95, 120}

auto fast = std::find_if(speeds.begin(), speeds.end(),
    [](int s) { return s > 100; });
if (fast != speeds.end()) {
    cout << "First speed over 100: " << *fast << endl;
}
```
**Key line to say:**
> "That lambda — `[](int s) { return s > 100; }` — is just a small nameless function you hand to the algorithm. You'll use this pattern constantly."

**Live demo — `count_if`, `transform`, `accumulate`:**
```cpp
#include <algorithm>
#include <numeric>

std::vector<int> speeds = {50, 120, 30, 95};

int fastCount = std::count_if(speeds.begin(), speeds.end(),
    [](int s) { return s > 60; });
cout << "Cars faster than 60: " << fastCount << endl;

std::vector<int> doubled(speeds.size());
std::transform(speeds.begin(), speeds.end(), doubled.begin(),
    [](int s) { return s * 2; });
// doubled is speeds with every value doubled

int total = std::accumulate(speeds.begin(), speeds.end(), 0);
cout << "Total speed: " << total << endl;
```
- Narrate each one: `count_if` counts matches, `transform` maps every element to a new value in another container, `accumulate` folds a range down to a single value (sum by default, but you can pass a custom combining function too).

**Sorting with smart pointers — a wrinkle worth flagging:**
```cpp
std::vector<std::unique_ptr<Car>> cars;
// ... fill cars ...
std::sort(cars.begin(), cars.end(),
    [](const std::unique_ptr<Car>& a, const std::unique_ptr<Car>& b) {
        return a->getSpeed() < b->getSpeed();
    });
```
**Key line to say:**
> "Sorting a vector of `unique_ptr`s works fine — `sort` moves the pointers themselves around, not the `Car` objects they point to, so ownership is preserved. You just need your comparison lambda to dereference (`->`) to compare the actual objects."

**Quick practice (5 min):**
> "Given `std::vector<std::unique_ptr<Item>> inventory`, write one line using `std::find_if` that finds the first item whose `getValue()` is over 50."

---

## 7. Wrap-up / Q&A (5 min)

**Recap out loud, in order:**
1. **RAII** — tie resource lifetime to object lifetime; acquire in the constructor, release in the destructor. Survives exceptions, unlike manual `delete`.
2. **Smart pointers** — `unique_ptr` for single ownership (default choice), `shared_ptr` for shared ownership, `weak_ptr` to observe without owning (breaks reference cycles)
3. **Move semantics** — `std::move` transfers ownership instead of copying; this is how `unique_ptr` and STL containers stay efficient
4. **STL containers** — `vector`, `map`, `unordered_map`, `set` hold your data (and can hold smart pointers, chaining RAII automatically)
5. **STL algorithms** — `sort`, `find_if`, `count_if`, `transform`, `accumulate` replace hand-written loops

**Tease next lesson:**
> "Next time: templates and generic programming — how the STL itself is built, and how to write your own reusable, type-flexible classes and functions."

---

## 8. The Project: Text-Based Dungeon Crawler (25 min: intro + starter skeleton)

This is where RAII, smart pointers, move semantics, and STL stop being separate topics and become one toolkit you reach for together.

**The pitch:**
You'll build a small dungeon crawler where a `Player` moves between `Room`s, picks up `Item`s, and fights `Enemy` objects. It's deliberately built so you *can't* avoid the concepts from today:

- **Polymorphism** — an `Item` base class with `Weapon`, `Potion`, `Armor` subclasses; an `Enemy` base class with different monster types. (If polymorphism and virtual functions are new to you, the short version: a base class can declare a method like `virtual void use() = 0;` with no body, forcing every subclass to provide its own — and calling `use()` on a base-class pointer or reference automatically runs the correct subclass's version. The skeleton below uses this directly.)
- **Ownership decisions (RAII/smart pointers)** — when a `Player` picks up an `Item` from a `Room`, ownership needs to actually transfer. Should that item live in `Room`'s container or `Player`'s? What happens to it in the old container? This is `std::move` and `unique_ptr` in a real scenario, not a toy one.
- **Move semantics** — the actual "pick up" operation is a `std::move` out of one `vector<unique_ptr<Item>>` and into another.
- **STL containers everywhere** — `Player` holds a `std::vector<std::unique_ptr<Item>>` inventory; `Room` holds items, enemies, and connections to other rooms (`std::map<std::string, Room*>` works well for "north/south/east/west" style exits, or `unordered_map` if you don't need exits listed in a consistent order).
- **STL algorithms** — searching your inventory for a specific item (`std::find_if`), sorting by strength/value (`std::sort`), counting how many potions you're carrying (`std::count_if`).

**Starter skeleton — live-code this together to kick the project off:**
```cpp
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class Item {
protected:
    string name;
public:
    Item(string n) : name(move(n)) {}
    virtual ~Item() { cout << name << " destroyed." << endl; }
    virtual void use() = 0;
    string getName() const { return name; }
};

class Potion : public Item {
public:
    Potion(string n) : Item(move(n)) {}
    void use() override { cout << "Drank " << name << ". Feeling better!" << endl; }
};

class Room {
private:
    vector<unique_ptr<Item>> items;
public:
    void addItem(unique_ptr<Item> item) {
        items.push_back(move(item));
    }

    // Removes and returns the named item from the room, or nullptr if not found.
    unique_ptr<Item> takeItem(const string& itemName) {
        auto it = find_if(items.begin(), items.end(),
            [&](const unique_ptr<Item>& i) { return i->getName() == itemName; });
        if (it == items.end()) return nullptr;

        unique_ptr<Item> found = move(*it);
        items.erase(it);
        return found;
    }
};

class Player {
private:
    vector<unique_ptr<Item>> inventory;
public:
    void pickUp(unique_ptr<Item> item) {
        cout << "Picked up " << item->getName() << "." << endl;
        inventory.push_back(move(item));
    }

    void listInventory() const {
        for (const auto& item : inventory) {
            cout << " - " << item->getName() << endl;
        }
    }
};

int main() {
    Room startRoom;
    startRoom.addItem(make_unique<Potion>("Health Potion"));

    Player player;
    unique_ptr<Item> picked = startRoom.takeItem("Health Potion");
    if (picked) {
        player.pickUp(move(picked));
    }

    player.listInventory();
}
```
**Walk through it out loud:**
> "Follow the `Health Potion` through its whole life: `make_unique` builds it inside `startRoom`'s vector. `takeItem` finds it with `find_if`, `move`s it out of the vector into a local variable, then `erase`s the now-empty slot. `pickUp` `move`s it again into the player's inventory. At every step there's exactly one `unique_ptr` owning that `Potion` — it's never copied, and it's never leaked, because ownership always has a clear, single home."

**Suggested build order (spread across future sessions):**
1. Get `Item` and its subclasses compiling with a `use()` virtual method — done above, extend with `Weapon` and `Armor`.
2. Get `Player` holding a `vector<unique_ptr<Item>>` and printing an inventory list — done above.
3. Get `Room` holding items, and the "pick up" logic that moves an item from `Room` to `Player` — done above.
4. Connect rooms together (`map<string, Room*>` for exits) and let the player navigate between them.
5. Add `Enemy` and basic turn-based combat using virtual `attack()`/`takeDamage()`.
6. Use `std::sort` to let the player view inventory sorted by value, and `std::count_if` to check "do I have a healing item?" before a fight.
7. Optional stretch goal: save/load the dungeon state to a text file.

**Leave them with a challenge:**
> "Before your next session, extend the skeleton above with a `Weapon` subclass and a second `Room` connected to the first by a `map<string, Room*>` exits table. Walk the player from Room A to Room B, and make sure a `Weapon` picked up in Room A shows up correctly in the player's inventory afterward — using `move`, not a copy. If you try to copy a `unique_ptr<Item>` instead, notice what the compiler tells you, and explain to yourself *why* it's right to stop you."

---

## Timing cheat sheet

| Section                                     | Minutes | Running total |
|----------------------------------------------|---------|----------------|
| Refresher: Pointers, `new`, Heap vs. Stack    | 10      | 10             |
| Why RAII?                                     | 10      | 20             |
| Smart Pointers (+ quick practice)             | 25      | 45             |
| Break                                         | 5       | 50             |
| Move Semantics                                | 15      | 65             |
| STL — Containers                              | 20      | 85             |
| STL — Algorithms (+ quick practice)           | 15      | 100            |
| Wrap-up / Q&A                                 | 5       | 105            |
| Project: intro + live-coded starter skeleton  | 15      | 120            |

**Running long?** Cut points, in order of safety:
1. Trim the `unordered_map`/`set` asides in Section 5 to one line each — mention them, don't demo them.
2. Drop the reference-cycle `weak_ptr` demo to a verbal explanation only; keep the `.lock()` one-liner.
3. Shorten the project section to just the pitch and build order — skip live-coding the starter skeleton and share it as a handout instead.

**Running short / audience is fast?** Add-ons that fit naturally:
- Have them predict `use_count()` values *before* running the `shared_ptr` demo instead of after.
- Ask them to spot the bug in a raw-pointer version of the dungeon-crawler `takeItem` function (hint: what if you `delete` the raw pointer both when removing it from `Room` and later when the `Player`'s inventory is destroyed?).
