# Dungeon Crawler Capstone — Staged Build Guide

This file explains each `.cpp` file in the staged build, in the order you'll live-code
them. Every stage compiles and runs on its own — none of them depend on a later file.
Each one builds directly on the stage before it, so diffing consecutive files is a good
way to see exactly what changed.

All stages compile with:

```zsh
g++ -std=c++17 -o <output_name> <filename>.cpp
```

---

## `stage1_navigation.cpp`

**Lesson section:** 2 — Building the World: Multi-Room Map & Navigation
**New concepts:** `map<string, Room*>`, raw non-owning pointers, `unique_ptr` ownership at the `main()` level

The smallest possible version of the game: three `Room`s (Entrance Hall, Armory, Damp
Cave) connected by exits, with `look`/`go <direction>`/`quit` commands. No items, no
enemies, no `Player` class yet.

`Room::exits` is a `map<string, Room*>` — a **raw pointer**, used deliberately. `main()`
owns every room in a `vector<unique_ptr<Room>>`, and since that vector outlives the
whole program, handing out raw `Room*` pointers via `.get()` is safe. This is the
baseline the later stages will complicate: Stage 5 revisits this exact design once
rooms need to be owned somewhere other than a single local variable.

**Try it:** `go north`, `go east`, `go west`, `go south` — walk the loop and notice you
can get back to where you started.

---

## `stage2_items.cpp`

**Lesson section:** 3 — Item Hierarchy: Weapon, Armor, Potion
**New concepts:** pure virtual functions, `unique_ptr<Item>`, forward declarations, `move`

Introduces the `Item` hierarchy: an abstract `Item` base class with `Weapon`, `Armor`,
and `Potion` subclasses, each overriding `use(Player&)` differently (equip vs. heal).
Also introduces a minimal `Player` who can `pickUp`, `findItem`, `removeItem`, and list
an inventory of `unique_ptr<Item>`.

Notice `class Player;` near the top — a **forward declaration**. `Item::use()` needs to
take a `Player&`, but `Player` itself needs to be defined after `Item` (since `Player`
doesn't depend on `Item`'s internals, only the reverse). The actual bodies of
`Potion::use()`, `Weapon::use()`, and `Armor::use()` are written *after* the full
`Player` class, once its public methods (`heal`, `equipWeapon`, `equipArmor`) exist to
call.

Navigation is gone in this stage — it returns in Stage 5. This stage is deliberately
narrowed to just items and the player, in a single fixed `Room`.

**Try it:** `take Iron Sword`, `use Iron Sword`, `inventory` — watch the equip message
fire, then `take Minor Potion`, `use Minor Potion` — watch it heal and then get
consumed (it disappears from `inventory` and prints its destructor message).

---

## `stage3_combat.cpp`

**Lesson section:** 4 — Enemy Hierarchy & Turn-Based Combat
**New concepts:** a second polymorphic hierarchy (`Enemy`/`Goblin`/`Dragon`), the
erase-remove idiom, `find_if`/`any_of` on `unique_ptr` containers

Adds `Enemy`, with `Goblin` (plain stats) and `Dragon` (overrides `attack()` to print a
flavor line). `Room` gains a `vector<unique_ptr<Enemy>>` alongside its items, plus
`firstLivingEnemy()` (`find_if`), `hasLivingEnemies()` (`any_of`), and
`clearDeadEnemies()` (`remove_if` + `erase` — the "erase-remove idiom").

`Player` gains `takeDamage()` (reduced by equipped armor's defense) and
`attackDamage()` (reads the equipped weapon's damage, or `5` for bare fists). A
standalone `fight(Player&, Room&)` function runs the turn loop: player hits enemy,
enemy hits back, repeat until one side is down.

**Try it:** the player starts pre-equipped with an Iron Sword. `fight` to take on the
Cave Goblin — watch the HP counters tick down each turn.

---

## `stage4_inventory_algorithms.cpp`

**Lesson section:** 6 — Inventory Management with STL Algorithms
**New concepts:** `sort`, `accumulate`, `count_if` on a `vector<unique_ptr<Item>>`

Same structure as Stage 3, with three new `Player` methods that put `<algorithm>` and
`<numeric>` to real use:

- `sortInventoryByValue()` — `sort` with a lambda comparator, highest value first
- `totalInventoryValue()` — `accumulate` folding the inventory down to a single sum
- `countPotions()` — `count_if` with a `dynamic_cast` predicate to count by subtype

New commands `sort` and `status` expose these. Every lambda takes
`const unique_ptr<Item>&` — never by value — since `unique_ptr` can't be copied; the
algorithms only ever look at inventory contents, they never take ownership of them.

**Try it:** `take` a few items of different values, `sort`, then `inventory` to see the
new order. `status` shows HP, total loot value, and potion count in one line each.

---

## `stage5_shared_weak_ptr.cpp`

**Lesson section:** 7 — The Room Graph Problem: shared_ptr, weak_ptr, and Cycles
**New concepts:** `shared_ptr`, `weak_ptr`, `.lock()`, reference cycles, a `World` class

The biggest structural change. Navigation returns, but `Room::exits` is no longer a raw
`Room*` — it's now `map<string, weak_ptr<Room>>`. A new `World` class owns every `Room`
via `vector<shared_ptr<Room>>`, which is the *only* place true ownership lives.

The demo world is a deliberate cycle: **Entrance → Armory → Cave → Entrance** (via a
"north" shortcut from the Cave straight back to the Entrance). This is the shape that
would leak if `exits` stored `shared_ptr` instead of `weak_ptr`: each room would hold a
`shared_ptr` to the next, the reference counts would never drop to zero even after
`World` itself is destroyed, and none of the three rooms would ever be freed. Storing
exits as `weak_ptr` and calling `.lock()` only when actually navigating avoids this —
the rooms have exactly one real owner (`World`), and everyone else just observes.

All the Stage 3/4 combat and inventory-algorithm code is carried over unchanged, now
routed through `World`'s wrapper methods (`go`, `take`, `useItem`, `fight`, `status`).

**Try it:** `go north` (Armory) → `go east` (Cave) → `go north` — you're back at the
Entrance Hall, having walked the full loop. Nothing leaks; nothing crashes.

---

## `stage6_full_game.cpp`

**Lesson section:** 8 — Full Game Loop Integration
**New concepts:** none — this is integration and polish

The finished game. Same architecture as Stage 5, expanded to the full four-room map
(Entrance Hall, Armory, Damp Cave, Dragon's Lair) with a real item/enemy layout, a
`help` command, and a win condition: defeating the `Dragon` in the Lair sets a
`victory` flag on `World`, and the main loop prints a final report (rooms visited via
the `set<string>`, total loot value via `accumulate`) before ending.

This file is identical in content to the `dungeon_crawler.cpp` delivered earlier —
`stage6` is just its name within the staged sequence, so the filenames read as one
continuous progression from Stage 1 through Stage 6.

**Try it:** play it start to finish — explore all four rooms, equip the Iron Sword and
Leather Armor from the Armory, defeat the Cave Goblin, then take on the Ancient Dragon
in the Lair.

---

## Quick reference: what's new at each stage

| Stage | Adds | Carries forward |
| --- | --- | --- |
| 1 | Rooms, raw-pointer exits, navigation | — |
| 2 | `Item` hierarchy, `Player`, inventory | (navigation dropped temporarily) |
| 3 | `Enemy` hierarchy, combat loop | Stage 2's items/Player |
| 4 | `sort`/`accumulate`/`count_if` on inventory | Stage 3's combat |
| 5 | `shared_ptr`/`weak_ptr` room graph, `World` class, navigation returns | Stages 2–4 combined |
| 6 | Win condition, `help`, final report | Everything |
