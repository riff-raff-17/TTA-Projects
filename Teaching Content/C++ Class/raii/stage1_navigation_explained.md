# Stage 1: Navigation — In-Depth Walkthrough

**File:** `stage1_navigation.cpp`
**Lesson section:** 2 — Building the World: Multi-Room Map & Navigation
**Compile:** `g++ -std=c++17 -o stage1_navigation stage1_navigation.cpp`

---

## What this stage is for

Before touching items, enemies, or combat, the game needs a *place* — a map you can
walk around in. This stage builds the smallest version of that: three rooms, connected
by named exits, with a text loop that lets you `look` and `go` between them.

Just as important as the map itself is the ownership decision baked into it: **who
actually owns each `Room` object, and who's just borrowing a pointer to one?** Getting
this right here, while the design is simple, sets up every later stage.

---

## The `Room` class

```cpp
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

    void describe() const { /* ... */ }
    const string& getName() const { return name; }
};
```

- **`exits` is a `map<string, Room*>`.** The key is a direction string ("north",
  "east", ...), the value is a *raw, non-owning pointer* to the neighboring room.
  Using `map` (rather than `unordered_map`) means iterating over `exits` in `describe()`
  always lists directions in alphabetical order — small, but predictable, which matters
  when you're demoing output live.
- **Why a raw pointer and not a `unique_ptr` or `shared_ptr`?** A `Room` doesn't *own*
  its neighbors — the Armory doesn't stop existing if the Entrance Hall does, and vice
  versa. Ownership and reference are different relationships, and `exits` is purely a
  reference. A raw pointer is the correct tool for "I need to refer to this object, but
  I am not responsible for its lifetime" — *as long as* something else guarantees the
  object outlives the pointer's use. That guarantee is what `main()` provides (see
  below). This is a deliberate, temporary design — Stage 5 revisits it once that
  guarantee gets harder to maintain.
- **`getExit()` returns `nullptr`** when the direction doesn't exist, rather than
  throwing or crashing. The caller (`main()`'s command loop) checks for `nullptr` and
  prints "You can't go that way." This null-check pattern — return a pointer that might
  be `nullptr`, and require the caller to check — shows up constantly in this codebase
  and is worth normalizing early.
- **The constructor uses `move(n)` and `move(desc)`.** Both parameters are passed by
  value (`string n`, `string desc`), so the caller's arguments are already copied once
  on the way in; using `move` to shuffle them into the member variables avoids a
  *second*, unnecessary copy. This is a common, idiomatic pattern for constructors that
  just want to "store" a passed-in string.

---

## Ownership in `main()`

```cpp
vector<unique_ptr<Room>> rooms;
rooms.push_back(make_unique<Room>("Entrance Hall", "A dusty stone hall. Torches flicker on the walls."));
rooms.push_back(make_unique<Room>("Armory", "Racks of rusted weapons line the walls."));
rooms.push_back(make_unique<Room>("Damp Cave", "Water drips somewhere in the darkness."));

rooms[0]->setExit("north", rooms[1].get());
rooms[1]->setExit("south", rooms[0].get());
rooms[1]->setExit("east", rooms[2].get());
rooms[2]->setExit("west", rooms[1].get());
```

- **`vector<unique_ptr<Room>> rooms`** is the single true owner of every `Room` in the
  program. Each `Room` has exactly one `unique_ptr` pointing at it, and that
  `unique_ptr` lives inside this vector for the entire run of `main()`.
- **`rooms[1].get()`** extracts the *raw* pointer out of the `unique_ptr` without
  transferring or sharing ownership. `.get()` never changes who owns the object — it
  just hands out a temporary, non-owning view of it. This is exactly the kind of raw
  pointer `Room::setExit()` expects.
- **Why this is safe:** `rooms` is a local variable in `main()`, and it isn't destroyed
  until `main()` returns — which is also when the whole program ends. Every raw
  `Room*` handed out via `.get()` is used only while `rooms` is still alive, so there's
  no way to end up with a "dangling" pointer (a pointer to an already-destroyed
  object). If `rooms` were destroyed early — say, if it were a local variable inside a
  function that returned before the game loop finished — every exit pointer would
  become dangerous to use.
- **The map has a loop already**, even at this small scale: Entrance ↔ Armory ↔ Cave
  is a chain, not a cycle, so there's no problem yet. But notice how easy it would be to
  add `rooms[2]->setExit("south", rooms[0].get())` and create one. Keep that thought —
  it's exactly the shape Stage 5 deals with.

---

## The command loop

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

    if (cmd == "quit") { break; }
    else if (cmd == "look") { current->describe(); }
    else if (cmd == "go") {
        Room* next = current->getExit(rest);
        if (!next) cout << "You can't go that way." << endl;
        else { current = next; /* ... */ }
    }
    else { cout << "Unknown command." << endl; }
}
```

- **Parsing:** `iss >> cmd` reads the first whitespace-delimited token (the command
  word). `getline(iss, rest)` then reads *everything remaining* on the line into
  `rest`, which is what lets multi-word arguments work later (e.g. "take Iron Sword" in
  Stage 2) — a plain `iss >> arg` would only capture "Iron", not "Iron Sword". The
  `.front() == ' '` check strips the one leading space left over between the command
  word and its argument.
- **`current` is a raw `Room*`** — again safe, because it always points somewhere
  inside `rooms`, which outlives the whole loop.
- **`if (!getline(cin, line)) break;`** exits cleanly when input runs out (e.g. when
  piping commands from a file or a script, as used for testing), not just when the
  user types `quit`.

---

## Key concepts this stage teaches

1. **Ownership vs. reference.** `rooms` owns; `exits` and `current` merely refer. Not
   every pointer needs to be a smart pointer — only the one(s) responsible for
   deletion do.
2. **`.get()`** is how you hand out a temporary, non-owning view of something a smart
   pointer owns, without disturbing that ownership.
3. **Lifetime reasoning.** A raw pointer is safe exactly as long as you can prove the
   thing it points to outlives every use of the pointer. Here, that proof is simple
   (`rooms` lives for all of `main()`) — Stage 5 is what happens when it stops being
   simple.

## Try it

- `go north`, `go east`, `go west`, `go south` — walk the full loop back to where you
  started.
- `go up` — see the "You can't go that way" message fire from a `nullptr` return.
- Add a fourth room and a fourth exit yourself; rebuild and confirm you can reach it.
