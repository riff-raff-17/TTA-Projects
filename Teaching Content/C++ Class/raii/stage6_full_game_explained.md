# Stage 6: Full Game — In-Depth Walkthrough

**File:** `stage6_full_game.cpp` (identical content to `dungeon_crawler.cpp`)
**Lesson section:** 8 — Full Game Loop Integration
**Compile:** `g++ -std=c++17 -o stage6_full_game stage6_full_game.cpp`

---

## What this stage is for

No new C++ concepts appear in this stage — every class (`Item`/`Weapon`/`Armor`/
`Potion`, `Enemy`/`Goblin`/`Dragon`, `Player`, `Room`, `World`) is exactly what Stages
2 through 5 built, unchanged. This stage is about **integration**: turning the pieces
into a complete, winnable game with a real four-room map, a help command, and an
ending. If Stages 1–5 were about introducing one idea at a time in isolation, this
stage is where you show that they all cooperate correctly under normal play.

---

## The win condition

```cpp
class World {
    // ...
    bool victory = false;
public:
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

    bool hasWon() const { return victory; }
};
```

- **`bool wasDragon = dynamic_cast<Dragon*>(enemy) != nullptr;` is captured *before*
  the fight loop runs**, not after. This ordering matters: by the time the fight ends,
  if the player won, `currentRoom->clearDeadEnemies()` is about to run and will destroy
  the `Dragon` object — trying to `dynamic_cast` it *after* that point would be
  operating on a dangling pointer. Checking the type up front, while `enemy` is still
  guaranteed valid, sidesteps that entirely.
- **`victory` is only set on a win, never reset.** Once true, it stays true — `World`
  doesn't need a way to "un-win," since the game ends on the first victory, and
  `main()`'s loop checks `hasWon()` right after every `fight` command and `break`s out
  as soon as it's true.
- **This is the same `dynamic_cast`-as-runtime-type-check pattern from Stage 4's
  `countPotions()`**, applied here to trigger a one-time event (ending the game)
  instead of counting.

---

## The finished room graph

```cpp
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
```

This is a denser version of Stage 5's three-room loop: four rooms, and *two* separate
paths between the Entrance and the Lair (the long way around through Armory/Cave, and
a direct "east"/"west" shortcut). It's still exactly the same `weak_ptr`-exit design
from Stage 5 — the extra rooms and extra loop don't require any new pointer machinery,
which is itself worth calling out: the fix from Stage 5 scales to an arbitrarily
complex map for free, because it was never about the *shape* of the graph, only about
which direction ownership flows.

The item and enemy placement is deliberate:
```cpp
entrance->addItem(make_unique<Potion>("Minor Potion", 10, 20));
armory->addItem(make_unique<Weapon>("Iron Sword", 50, 15));
armory->addItem(make_unique<Armor>("Leather Armor", 40, 5));
cave->addItem(make_unique<Potion>("Cave Potion", 15, 15));
cave->addEnemy(make_unique<Goblin>("Cave Goblin"));
lair->addItem(make_unique<Weapon>("Dragon Slayer", 200, 30));
lair->addEnemy(make_unique<Dragon>("Ancient Dragon"));
```
The Armory sits between the entrance and any danger, so a careful player naturally
gears up (sword + armor) before reaching the Cave Goblin, and the Dragon Slayer — a
strictly better weapon than the Iron Sword — sits in the same room as the final boss,
available whether or not the player thinks to grab it before or after the fight.

---

## The command loop, in full

```cpp
istringstream iss(line);
string cmd;
iss >> cmd;
string rest;
getline(iss, rest);
if (!rest.empty() && rest.front() == ' ') rest.erase(0, 1);

if (cmd == "quit" || cmd == "exit") { /* ... */ break; }
else if (cmd == "help") { printHelp(); }
else if (cmd == "look") { world.look(); }
else if (cmd == "go") { world.go(rest); }
else if (cmd == "take") { world.take(rest); }
else if (cmd == "inventory" || cmd == "i") { world.getPlayer().listInventory(); }
else if (cmd == "sort") { world.getPlayer().sortInventoryByValue(); }
else if (cmd == "use") { world.useItem(rest); }
else if (cmd == "fight") { world.fight(); }
else if (cmd == "status") { world.status(); }
else if (cmd.empty()) { /* ignore blank input */ }
else { cout << "Unknown command. Type 'help'." << endl; }

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
```

- **Every command handler is now a one-line call into `World`** (`world.go(rest)`,
  `world.fight()`, etc.) — `main()` itself contains no game logic anymore, only
  parsing and dispatch. This is the payoff of Stage 5's refactor: once `World` existed
  as a proper owner of game state, `main()`'s job shrank down to "read a line, figure
  out which `World` method that maps to, call it."
- **`cmd == "inventory" || cmd == "i"`** — a short alias, purely a usability nicety for
  players who'll type this command often.
- **The two post-command checks (`isAlive()`, `hasWon()`) run after *every* command**,
  not just after `fight`. Death can only actually happen as a result of `fight`, but
  checking unconditionally after every command is simpler to read than threading a
  "did that command just end the game?" flag through every branch above — a reasonable
  simplicity/efficiency tradeoff for a text loop that runs a handful of times a second
  at most.
- **The final report reuses Stage 4's `totalInventoryValue()`** and `World`'s own
  `visitedRooms.size()` (exposed via `roomsVisited()`) — nothing new is computed here,
  it's simply the STL-algorithm-driven stats from Stage 4, surfaced one more time at
  the moment they matter most.

## Key concepts this stage teaches

1. **Integration is its own skill.** Nothing here is a new C++ feature — it's applying
   everything from Stages 1–5 together and confirming the pieces don't conflict.
2. **Capturing type information before a destructive operation** (`wasDragon` checked
   before `clearDeadEnemies()` runs) — a general pattern any time you need to know
   something about an object that's about to be destroyed.
3. **A "thin" `main()`** — once enough logic has moved into properly owned classes
   (`World`, `Player`, `Room`), the entry point's job reduces to parsing input and
   dispatching, which is a healthy sign the ownership design from earlier stages was
   sound.

## Try it

- Play the game to completion: gear up in the Armory, clear the Cave Goblin, then
  defeat the Ancient Dragon in the Lair, and read the final report.
- Try reaching the Lair via the direct Entrance → east shortcut instead of the long way
  around through Armory and Cave — confirm the win condition still fires correctly
  regardless of path taken.
- Try `fight` in a room with no living enemies (e.g. the Entrance Hall) — confirm
  `firstLivingEnemy()` correctly returns `nullptr` and prints "Nothing here to fight."
  without crashing.
