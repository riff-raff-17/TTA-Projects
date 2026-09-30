# Stage 3: Combat — In-Depth Walkthrough

**File:** `stage3_combat.cpp`
**Lesson section:** 4 — Enemy Hierarchy & Turn-Based Combat
**Compile:** `g++ -std=c++17 -o stage3_combat stage3_combat.cpp`

---

## What this stage is for

Stage 2 gave the game things to pick up; this stage gives it something to fight. It
introduces a second polymorphic hierarchy (`Enemy`) alongside `Item`, adds enemies to
`Room`, and adds combat-related methods to `Player`. It's also the first stage where
smart-pointer-owned objects go through repeated, stateful changes across multiple
turns — a good test that the ownership model from Stage 2 holds up under more than a
single pickup-and-use.

Everything from Stage 2 (`Item`, `Weapon`, `Armor`, `Potion`, the core of `Player`) is
carried over unchanged; this walkthrough only covers what's new.

---

## The `Enemy` hierarchy

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

    void takeDamage(int dmg) {
        health -= dmg;
        if (health < 0) health = 0;
        cout << "  " << name << " takes " << dmg << " damage (" << health << " HP left)." << endl;
    }

    virtual int attack() const { return attackPower; }
    virtual string describe() const { /* ... */ }
    const string& getName() const { return name; }
};
```

- **Structurally this mirrors `Item`**: a base class with shared state (`name`,
  `health`, `attackPower`), a virtual destructor (for the same reason as `Item`'s —
  `Enemy` objects are destroyed through base-class pointers, e.g. when a
  `unique_ptr<Enemy>` in `Room::enemies` is erased), and virtual methods subclasses can
  override.
- **`attack()` is virtual but *not* pure virtual** (no `= 0`). Unlike `Item::use()`,
  which had no sensible default, every `Enemy` has a reasonable default attack: just
  deal `attackPower` damage. `Goblin` doesn't override it at all — it inherits the base
  behavior unchanged. `Dragon` overrides it to print a flavor line first. This is the
  distinction between "every subclass must define this from scratch" (pure virtual)
  and "there's a sensible default, but subclasses may customize it" (plain virtual).
- **`takeDamage()` clamps at zero**: `if (health < 0) health = 0;`. Without this, an
  overkill hit (say, 15 damage into 5 remaining HP) would leave `health` at `-10`,
  which would still satisfy `isAlive()` returning false correctly (since the check is
  `health > 0`), but would print a confusing negative HP value. Clamping keeps the
  displayed number sensible without changing the alive/dead logic.

```cpp
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
```

- **Default constructor arguments** (`string n = "Goblin"`) mean you can write
  `make_unique<Goblin>()` for a generically-named goblin, or
  `make_unique<Goblin>("Cave Goblin")` to give it a specific name — both compile, and
  the flavor text stays flexible without needing two separate constructors.
- **`explicit`** on a single-argument constructor prevents the compiler from silently
  using it for implicit conversions (e.g., accidentally allowing a bare string to
  convert into a `Goblin` somewhere you didn't intend one). It's a defensive habit for
  any constructor that can be called with one argument.
- **`Dragon::attack()` calling the base `attackPower`** shows overriding doesn't have
  to mean *replacing* the base behavior entirely — it added a print statement and then
  fell back to the same damage calculation as any other `Enemy`.

---

## `Room` gains enemies

```cpp
class Room {
private:
    string name;
    vector<unique_ptr<Item>> items;
    vector<unique_ptr<Enemy>> enemies;
public:
    void addEnemy(unique_ptr<Enemy> enemy) { enemies.push_back(move(enemy)); }

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
};
```

- **`enemies` is a second `vector<unique_ptr<...>>` alongside `items`** — same
  ownership pattern, different element type. `Room` is the sole owner of every enemy
  placed in it.
- **`any_of`** answers a yes/no question over a range: "does at least one element
  satisfy this predicate?" Here, "is there at least one living enemy?" It short-circuits
  — stops checking as soon as it finds one match — so it's efficient even with a large
  `enemies` vector.
- **`find_if` returns an *iterator***, not a value — `it == enemies.end()` is how you
  check "did we find nothing?" (an iterator equal to `end()` means the search reached
  the end of the range without matching). `it->get()` then extracts the raw `Enemy*`
  from the `unique_ptr` at that position.
- **`clearDeadEnemies()` uses the erase-remove idiom**, one of the most common STL
  patterns and worth naming explicitly the first time you see it:
  - `remove_if` doesn't actually shrink the vector. It rearranges elements so that all
    the ones matching the predicate (`!e->isAlive()`, i.e. dead ones) are moved to the
    *end*, and returns an iterator marking where the "should be removed" section
    begins. The vector's size is unchanged at this point — the "removed" elements are
    still technically present, just in an unspecified state at the tail end.
  - `enemies.erase(new_end, enemies.end())` then actually deletes that tail section,
    which is where the dead `Enemy` objects' destructors run (printing "is gone")
    and the vector actually shrinks.
  - The two-step split (`remove_if` rearranges, `erase` deletes) is *why* it's called
    "erase-remove" — the vector's own `erase()` alone doesn't know how to search for
    matching elements, and `remove_if` alone doesn't know how to actually shrink a
    container. Combined, they do both.

---

## `Player` gains combat stats

```cpp
void takeDamage(int dmg) {
    int reduced = equippedArmor ? max(0, dmg - equippedArmor->getDefense()) : dmg;
    health -= reduced;
    if (health < 0) health = 0;
    cout << name << " takes " << reduced << " damage (" << health << "/" << maxHealth << " HP)." << endl;
}

int attackDamage() const { return equippedWeapon ? equippedWeapon->getDamage() : 5; }
bool isAlive() const { return health > 0; }
```

- **`takeDamage` reads `equippedArmor`'s defense value if it's set.** This is the first
  place Stage 2's non-owning `equippedArmor` pointer actually pays off gameplay-wise:
  `Player` doesn't need to duplicate armor data anywhere, it just asks the object
  `inventory` already owns.
  `max(0, dmg - equippedArmor->getDefense())` prevents armor from ever turning damage
  negative (which would heal the player on a hit, clearly wrong).
- **The ternary `equippedArmor ? ... : dmg`** checks the pointer itself for
  truthiness — a raw pointer is implicitly convertible to `bool`, `true` if non-null.
  This is the standard way to branch on "is there something equipped or not?" without
  writing out `equippedArmor != nullptr` every time (though that's equivalent and
  sometimes preferred for clarity).
- **`attackDamage()` follows the same pattern for the weapon**, with `5` as a fallback
  representing bare-fisted combat.

---

## The `fight()` function

```cpp
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
```

- **This is a free function, not a method on `Player` or `Room`** — it takes both by
  reference because combat genuinely involves two objects interacting, and neither one
  is a more natural "owner" of the fight logic than the other. (Stage 5's `World` class
  will eventually wrap this as a method, once there's a natural single owner of "the
  current game state" to attach it to.)
- **The `while` loop alternates turns.** Player always attacks first each turn; the
  `if (!enemy->isAlive()) break;` check after the player's hit prevents a defeated
  enemy from getting a free retaliation hit in — once it's dead, the loop exits before
  `enemy->attack()` is ever called.
- **`enemy->attack()`** is a virtual call — for a `Goblin` this just returns
  `attackPower`; for a `Dragon` it also prints the fire-breath line first. `fight()`
  doesn't need to know or care which — that's the whole benefit of polymorphism showing
  up in the control flow itself.
- **After combat, `room.clearDeadEnemies()`** actually removes the (now dead) enemy
  from the room, running its destructor. Notice this only happens on a *win* — if the
  player dies, the (still-alive) enemy correctly stays in the room.

## Key concepts this stage teaches

1. **Plain virtual vs. pure virtual**: `Enemy::attack()` shows the "sensible default,
   optional override" pattern, contrasting with `Item::use()`'s "no default, must
   override" from Stage 2.
2. **The erase-remove idiom** (`remove_if` + `erase`) — the standard way to conditionally
   delete elements from a `vector`.
3. **`any_of`** for yes/no questions over a range, as a companion to `find_if` for
   "find the first match."
4. **Raw observer pointers powering real gameplay logic** — `equippedArmor` and
   `equippedWeapon` aren't just stored, they're read every single combat turn.

## Try it

- `fight` immediately — the player starts pre-equipped with an Iron Sword; watch the
  HP counters tick down turn by turn.
- Remove the pre-equip line and `fight` bare-fisted — confirm `attackDamage()` falls
  back to `5` and the fight takes longer.
- Add a second `Goblin` to the room and call `fight` twice — confirm
  `firstLivingEnemy()` correctly finds the next one once the first is cleared.
