# Stage 5: shared_ptr / weak_ptr Room Graph — In-Depth Walkthrough

**File:** `stage5_shared_weak_ptr.cpp`
**Lesson section:** 7 — The Room Graph Problem: shared_ptr, weak_ptr, and Cycles
**Compile:** `g++ -std=c++17 -o stage5_shared_weak_ptr stage5_shared_weak_ptr.cpp`

---

## What this stage is for

This is the biggest structural change in the whole staged build. Every prior stage's
`Item`, `Enemy`, and `Player` code carries over completely unchanged — what's new is
*how rooms own each other*, and a new `World` class that gives ownership a proper home.
This stage exists to answer the question planted all the way back in Stage 1: what
happens once a raw pointer's safety guarantee (Stage 1's `rooms` vector outliving
everything) stops being simple to maintain?

---

## Why Stage 1's design needed to change

Stage 1's `Room::exits` was `map<string, Room*>` — safe *only* because `main()` kept
every `Room` alive in a `vector<unique_ptr<Room>>` for the program's entire run. That
worked because there was exactly one place responsible for room lifetimes, and it
never went away early.

But consider a design where ownership needs to be shared — say, a `World` that owns
rooms, plus some other system (a save manager, a quest tracker, anything) that also
needs to keep a room alive independently. Once two different parts of a program each
want to hold real ownership, the natural reach is `shared_ptr` on both sides. And
that's exactly where a **room graph** causes trouble: a dungeon map almost always has
loops in it (there's often more than one way back to a room you've already visited).
If two rooms hold `shared_ptr`s to each other as part of that loop, their reference
counts keep each other above zero *forever*, even after every other part of the
program has let go. Neither room's destructor ever runs. No crash, no warning — just
memory that's silently never reclaimed.

---

## The fix: one real owner, everyone else observes

```cpp
class World {
private:
    vector<shared_ptr<Room>> rooms; // the one true owner of every room
    // ...
};

class Room {
private:
    map<string, weak_ptr<Room>> exits; // observes other rooms without owning them
    // ...
};
```

- **`World::rooms` is a `vector<shared_ptr<Room>>`.** This is the *only* place genuine
  ownership lives. As long as `World` exists, every room it created stays alive —
  exactly the same guarantee Stage 1's `main()`-owned vector provided, just now
  encapsulated inside a proper class instead of a local variable.
- **`Room::exits` is `map<string, weak_ptr<Room>>`, not `shared_ptr`.** A `weak_ptr`
  can *observe* an object that a `shared_ptr` elsewhere owns, without adding to its
  reference count. It answers "is this object still alive, and if so, can I get a
  usable pointer to it?" — but it never keeps the object alive by itself. This is what
  breaks the cycle: Entrance can point at Armory, and Armory can point right back at
  Entrance, and neither reference count goes up because of it. Only `World::rooms`
  counts.

```cpp
void setExit(const string& direction, const shared_ptr<Room>& room) {
    exits[direction] = room; // stored as weak_ptr — does not extend room's lifetime
}

shared_ptr<Room> getExit(const string& direction) const {
    auto it = exits.find(direction);
    if (it == exits.end()) return nullptr;
    return it->second.lock(); // nullptr if the room somehow no longer exists
}
```

- **`setExit` takes a `const shared_ptr<Room>&`** as its parameter — a `shared_ptr` is
  the natural type to pass in, since that's what `World::addRoom()` hands back. But the
  assignment `exits[direction] = room;` is assigning into a `weak_ptr` map, so the
  `shared_ptr` argument is implicitly converted to a `weak_ptr` at that point — this
  conversion exists specifically to support this pattern (constructing/assigning a
  `weak_ptr` from a `shared_ptr`), and it does not touch the reference count upward.
- **`getExit` calls `.lock()` on the stored `weak_ptr`.** This is the only way to
  actually *use* a `weak_ptr` — you can't dereference it directly. `.lock()` checks
  whether the object is still alive (i.e., some `shared_ptr` still owns it) and, if so,
  hands back a temporary, fully-functional `shared_ptr` to it (which *does* bump the
  reference count, for as long as that temporary lives). If the object is already gone,
  `.lock()` returns an empty `shared_ptr`, which converts to `nullptr` in a boolean or
  pointer-comparison context. In this program, rooms never actually go away, so `.lock()`
  always succeeds — but the pattern is worth understanding for any case where the
  observed object's lifetime genuinely isn't guaranteed.

---

## The `World` class in full

```cpp
class World {
private:
    vector<shared_ptr<Room>> rooms;
    shared_ptr<Room> currentRoom;
    Player player;
    set<string> visitedRooms;

public:
    shared_ptr<Room> addRoom(const string& name, const string& desc) {
        auto room = make_shared<Room>(name, desc);
        rooms.push_back(room);
        return room;
    }
    // go(), take(), useItem(), fight(), status() — all wrap the Stage 3/4 logic
};
```

- **`addRoom` uses `make_shared<Room>(...)`**, the `shared_ptr` equivalent of
  `make_unique`. It both allocates the `Room` and wraps it in a `shared_ptr` in one
  step, which is more efficient than separately calling `new Room(...)` and then
  constructing a `shared_ptr` from that raw pointer (the combined form allocates the
  control block — the piece that tracks the reference count — together with the object
  itself, in a single allocation).
- **`addRoom` both stores the `shared_ptr` in `rooms` *and* returns a copy of it.**
  That's not a contradiction — copying a `shared_ptr` is exactly what it's designed
  for; it bumps the reference count and both copies point at the same object. This is
  why `main()` can do `auto entrance = world.addRoom(...)` and get back a `shared_ptr`
  it can immediately use to call `setExit()` on other rooms, while `World::rooms` keeps
  its own independent copy as the "real" owner.
- **`currentRoom` is also a `shared_ptr<Room>`**, not a raw pointer like Stage 1's
  `current`. Since rooms are now owned via `shared_ptr` throughout the class, it's
  natural (and safe) for `World` to hold its "where the player currently is" pointer
  the same way — though a raw pointer *would* also have been safe here, since
  `currentRoom` never outlives `World` itself. Using `shared_ptr` here is a matter of
  consistency with the rest of the class more than strict necessity.
- **`Player player;`** — note this is a *value* member, not a pointer at all. `World`
  owns exactly one `Player`, directly, with no dynamic allocation involved. Not every
  piece of state needs a pointer — only things whose lifetime needs to be independently
  managed, shared, or potentially absent (`nullptr`-able) benefit from one.

```cpp
void go(const string& direction) {
    if (direction.empty()) { cout << "Go where?" << endl; return; }
    auto next = currentRoom->getExit(direction);
    if (!next) { cout << "You can't go that way." << endl; return; }
    currentRoom = next;
    visitedRooms.insert(currentRoom->getName());
    cout << "You head " << direction << "." << endl;
    currentRoom->describe();
}
```

- **`auto next = currentRoom->getExit(direction);`** — `next`'s type is deduced as
  `shared_ptr<Room>` (that's `getExit`'s return type). If the exit didn't exist, or if
  `.lock()` somehow failed, `next` would be an empty `shared_ptr`, which the
  `if (!next)` check catches — same null-check pattern as Stage 1's raw pointer
  version, just now on a smart pointer instead.
- **`currentRoom = next;`** reassigns which `shared_ptr` `currentRoom` holds — the old
  room's reference count drops by one (though it stays alive regardless, since
  `World::rooms` still owns it), and the new room's count goes up by one for as long as
  `currentRoom` points at it.

---

## The demo map: an actual cycle

```cpp
auto entrance = world.addRoom("Entrance Hall", "A dusty stone hall.");
auto armory   = world.addRoom("Armory", "Racks of rusted weapons line the walls.");
auto cave     = world.addRoom("Damp Cave", "Water drips somewhere in the darkness.");

entrance->setExit("north", armory);
armory->setExit("south", entrance);
armory->setExit("east", cave);
cave->setExit("west", armory);
cave->setExit("north", entrance); // the shortcut that completes the loop
```

Walk through the cycle explicitly: Entrance → (north) → Armory → (east) → Cave →
(north) → back to Entrance. If `setExit` stored `shared_ptr` instead of `weak_ptr`,
this alone would be enough to leak all three rooms — Entrance would hold a `shared_ptr`
to Armory, Armory to Cave, Cave back to Entrance, and even after `World` (and its
`rooms` vector) is destroyed at the end of `main()`, each room's reference count would
still be propped up by the next room in the loop pointing at it. Because `exits` is
`weak_ptr`, none of that ownership exists — only `World::rooms` counts, and when
`World` is destroyed, all three rooms are destroyed cleanly with it.

## Key concepts this stage teaches

1. **`shared_ptr` for genuine shared ownership**, `weak_ptr` for observing without
   owning — the pairing only makes sense together; a `weak_ptr` is meaningless without
   some `shared_ptr` elsewhere actually owning the object.
2. **`.lock()`** is the only way to use a `weak_ptr`'s target, and it can fail (return
   an empty `shared_ptr`), which callers must check for.
3. **Reference cycles are a real, silent leak risk** whenever a graph-shaped ownership
   structure (rooms with exits back to each other, parent/child relationships, etc.)
   uses `shared_ptr` on every edge. The fix is always the same shape: pick one
   direction as "real" ownership, make the other direction observe-only.
4. **`make_shared`** is the preferred way to construct a `shared_ptr`-managed object,
   for the same reason `make_unique` is preferred for `unique_ptr` — fewer manual `new`
   calls, and (for `shared_ptr` specifically) a single combined allocation.

## Try it

- `go north`, `go east`, `go north` — walk the full loop back to the Entrance Hall and
  confirm nothing crashes or behaves oddly.
- Mentally (or with a debugger) trace what `World`'s destructor does at the end of
  `main()`: `rooms` is destroyed, which destroys each `shared_ptr<Room>`, which (since
  no `weak_ptr` counts toward the reference count) immediately drops each room's count
  to zero and destroys it — regardless of the loop in `exits`.
- As a thought experiment: what would need to change if you wanted a `Quest` object
  that a `Room` could reference, where the `Quest` is owned elsewhere (say, by
  `World`)? (Answer: the same pattern — `World` holds `shared_ptr<Quest>`, `Room` holds
  `weak_ptr<Quest>`.)
