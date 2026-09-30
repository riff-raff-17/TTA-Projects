# Stage 2: Items — In-Depth Walkthrough

**File:** `stage2_items.cpp`
**Lesson section:** 3 — Item Hierarchy: Weapon, Armor, Potion
**Compile:** `g++ -std=c++17 -o stage2_items stage2_items.cpp`

---

## What this stage is for

This stage introduces the game's first polymorphic hierarchy — `Item` and its three
subclasses — and the `Player` who carries them. Navigation is intentionally dropped
for this stage (it returns in Stage 5) so the focus stays entirely on ownership and
polymorphism: how a `unique_ptr<Item>` moves between containers, and how calling one
method (`use()`) can mean three different things depending on what's actually being
used.

---

## The forward declaration

```cpp
class Player; // forward declaration — Item::use() needs a reference to Player
```

`Item` declares a method `virtual void use(Player& player) = 0;`, which means the
compiler needs to know `Player` *exists* (so it can form a reference type `Player&`)
before it's fully defined. A forward declaration — just the class name and a
semicolon — is enough for that; the compiler doesn't need to know `Player`'s members
yet, only that the type exists. `Player`'s full definition comes later in the file,
once `Item` and its subclasses are done.

---

## The `Item` hierarchy

```cpp
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
```

- **`virtual ~Item()`** — a virtual destructor is required whenever a class is meant to
  be used polymorphically (i.e., deleted through a base-class pointer, which is exactly
  what happens when a `unique_ptr<Item>` holding a `Weapon` is destroyed). Without
  `virtual` here, destroying a `Weapon` through an `Item*` would only run `~Item()`,
  silently skipping `~Weapon()` — usually harmless here since `Weapon` has no extra
  resources to clean up, but it's the kind of bug that only shows up once a subclass
  *does* own something. Get in the habit of marking base-class destructors `virtual`
  whenever the class has any other virtual methods.
- **`use(Player&) = 0;` and `describe() const = 0;`** are *pure virtual* — the `= 0`
  means `Item` itself cannot be instantiated, and every concrete subclass **must**
  provide its own implementation. This is the compiler enforcing "every item type needs
  to define what happens when you use it" as a hard rule, not a convention you have to
  remember.
- **`protected` members** (`name`, `value`) are visible to `Item` and its subclasses,
  but not to outside code — `Weapon`, `Armor`, and `Potion` can read `name` and `value`
  directly in their constructors and `describe()` overrides, but `main()` has to go
  through `getName()`/`getValue()`.

### `Potion`, `Weapon`, `Armor`

```cpp
class Potion : public Item {
    int healAmount;
public:
    Potion(string n, int v, int heal) : Item(move(n), v), healAmount(heal) {}
    void use(Player& player) override;
    string describe() const override { /* ... */ }
};
```

Each subclass adds exactly one new stat (`healAmount`, `damage`, `defense`) and
overrides both pure virtual methods. The `use()` bodies are declared here but *defined
later*, after `Player` is complete — this is the same forward-declaration trick as
before, applied at the method level: the class can *promise* the method exists without
the compiler needing `Player`'s internals yet.

`override` isn't strictly required by the compiler, but it's a safety net: if you
mistype a signature (wrong parameter type, missing `const`, etc.), `override` turns
what would otherwise be a silent bug — you think you overrode the base method, but you
actually just added an unrelated new one — into a compile error.

---

## The `Player` class

```cpp
class Player {
private:
    string name;
    int health;
    int maxHealth;
    vector<unique_ptr<Item>> inventory;
    Weapon* equippedWeapon = nullptr; // non-owning observer into inventory
    Armor* equippedArmor = nullptr;   // non-owning observer into inventory
```

- **`inventory` is a `vector<unique_ptr<Item>>`.** Each item in it is owned by exactly
  this vector — nothing else in the program holds a `unique_ptr` to the same object at
  the same time (that's what `unique_ptr` enforces: it cannot be copied, only moved).
- **`equippedWeapon` and `equippedArmor` are raw pointers**, not `unique_ptr` or
  `shared_ptr`. This mirrors Stage 1's `Room::exits` design: the equipped weapon is
  *already* owned by `inventory`; `equippedWeapon` just remembers *which* inventory
  item is currently equipped, without owning a second copy of it. If `Player` tried to
  own it twice — once in `inventory`, once via a second `unique_ptr` — that would be a
  double-ownership bug, exactly the thing `unique_ptr`'s copy-prohibition exists to
  prevent.

```cpp
    void pickUp(unique_ptr<Item> item) {
        cout << "Picked up " << item->getName() << "." << endl;
        inventory.push_back(move(item));
    }
```

- **`pickUp` takes its parameter by value: `unique_ptr<Item> item`.** Since
  `unique_ptr` can't be copied, the *only* way to call this function is to pass
  something that's already being moved in — e.g. `player.pickUp(move(someItem))`, or
  the return value of a function like `Room::takeItem()`, which is already a temporary
  and moves in automatically. The signature itself documents "ownership transfers into
  this function" — you can't accidentally call it in a way that would leave two owners.
- **`move(item)` inside the function body** then moves that same `unique_ptr` into
  `inventory`. After this line, the local parameter `item` is empty (`nullptr`);
  `inventory`'s last element now owns the `Item`.

```cpp
    Item* findItem(const string& itemName) {
        auto it = find_if(inventory.begin(), inventory.end(),
            [&](const unique_ptr<Item>& i) { return i->getName() == itemName; });
        return it == inventory.end() ? nullptr : it->get();
    }
```

- **`find_if`** scans `inventory` for the first element matching a predicate — here, a
  lambda comparing names. The lambda parameter is `const unique_ptr<Item>&`: a
  reference, never a copy, because `unique_ptr` isn't copyable. `[&]` captures the
  enclosing `itemName` by reference so the lambda body can use it.
- **The return type is `Item*`**, a raw pointer, not `unique_ptr<Item>&` or anything
  that implies ownership. `findItem` is a *lookup*, not a transfer — the caller gets a
  temporary, non-owning view to read from or call methods on, and `inventory` keeps
  ownership throughout.

```cpp
    unique_ptr<Item> removeItem(const string& itemName) {
        auto it = find_if(/* ... */);
        if (it == inventory.end()) return nullptr;
        unique_ptr<Item> found = move(*it);
        inventory.erase(it);
        if (equippedWeapon == found.get()) equippedWeapon = nullptr;
        if (equippedArmor == found.get()) equippedArmor = nullptr;
        return found;
    }
```

- **This is different from `findItem`: `removeItem` actually transfers ownership out.**
  `move(*it)` moves the `unique_ptr` out of the vector slot (leaving that slot holding
  `nullptr`), then `inventory.erase(it)` removes the now-empty slot itself.
- **The equip-pointer cleanup is important.** If the item being removed happened to be
  the currently equipped weapon or armor, `equippedWeapon`/`equippedArmor` would become
  a *dangling pointer* — pointing at memory that's about to be freed — the instant the
  returned `unique_ptr<Item>` goes out of scope wherever the caller doesn't keep it
  alive. Setting them back to `nullptr` here prevents that. This is a small but
  realistic example of the bookkeeping raw observer pointers require: whoever removes
  the owned object is responsible for clearing out anyone still watching it.

---

## Wiring `use()` back to `Player`

```cpp
void Potion::use(Player& player) {
    cout << player.getName() << " drinks the " << name << "." << endl;
    player.heal(healAmount);
}
void Weapon::use(Player& player) { player.equipWeapon(this); }
void Armor::use(Player& player) { player.equipArmor(this); }
```

These three definitions appear *after* the full `Player` class — this is why the
forward declaration and the "declare now, define later" pattern from earlier were
necessary. Notice `player.equipWeapon(this)`: `this` inside `Weapon::use()` is the
`Weapon*` currently being used, which is exactly the raw pointer `Player` wants to
remember as "the equipped weapon." No new object is created and no ownership changes —
`inventory` still owns the `Weapon`; `Player` just now has a pointer to it.

---

## `main()`: the demo loop

```cpp
Item* item = player.findItem(rest);
if (!item) { cout << "You don't have that." << endl; continue; }
bool isPotion = dynamic_cast<Potion*>(item) != nullptr;
item->use(player);
if (isPotion) player.removeItem(rest);
```

- **`item->use(player)` is a polymorphic call.** The actual code that runs — heal, or
  equip — depends on the *dynamic type* of the object `item` points to, not on the
  static type (`Item*`) of the pointer itself. This is virtual dispatch: the same line
  of code produces three different behaviors depending on what's really being pointed
  to.
- **`dynamic_cast<Potion*>(item)`** is a runtime check: "is this object actually a
  `Potion`, specifically?" It's used here for one reason: potions are single-use and
  should be removed from inventory after drinking, while weapons and armor stay
  equipped and remain in the inventory list. Checking *before* calling `use()` and
  removing *after* avoids removing the item while `use()` is still running.

## Key concepts this stage teaches

1. **Pure virtual functions** enforce "every subclass must implement this" at compile
   time.
2. **`unique_ptr` parameters passed by value** are the idiomatic way to say "this
   function takes ownership."
3. **Raw pointers as non-owning observers** (`equippedWeapon`/`equippedArmor`) require
   manual cleanup when the owned object is removed — a cost worth knowing about before
   reaching for a raw pointer.
4. **`find_if` + lambda** is the standard STL pattern for "look up by some property," and
   always takes containers of `unique_ptr` by reference in the lambda signature.

## Try it

- `take Iron Sword`, `use Iron Sword`, `inventory` — confirm the sword shows as
  equipped and stays in the list.
- `take Minor Potion`, `use Minor Potion` — confirm it heals and then disappears from
  `inventory`, printing its destructor message.
- `use Leather Armor` before taking it — confirm `findItem` correctly reports "You
  don't have that."
