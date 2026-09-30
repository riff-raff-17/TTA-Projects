# Templates & Generic Programming in C++ — 2-Hour Lesson: Talking Points

This lesson is self-contained — every class and function used below is written from scratch, so no prior code file is required. It's the natural next step after a lesson on RAII, smart pointers, move semantics, and the STL: today opens up *how* the STL itself is built, and gives you the tools to write your own reusable, type-flexible classes and functions.

---

## 0. Setup (before class starts)

- Have a blank `.cpp` file ready, compiler set to C++17 or later. Section 7 briefly touches C++20 `concepts` — have a fallback ready (`static_assert`) if your compiler doesn't support them, and just mention concepts verbally if not.
- Assumes comfort with: classes, constructors/destructors, `std::unique_ptr`, `std::vector`/`std::map`, `std::move`, and lambdas (`[](...) {...}`).
- Plan a real 5-minute break around the halfway mark (after Class Templates).

---

## 1. Refresher: The Problem Templates Solve (10 min)

**Talking points:**

- You've already written near-duplicate functions before: one version of a helper for `int`, another for `double`, another for `std::string` — same logic, different type, copy-pasted three times.
- **A template lets you write the logic once, parameterized by type, and the compiler generates the type-specific version for you automatically — at compile time.** There's no runtime cost versus hand-writing each version yourself; the compiler literally produces a separate function or class per type used, a process called **instantiation**.
- This is exactly how `std::vector<T>`, `std::unique_ptr<T>`, `std::map<K,V>`, and `std::sort` all work with any type you throw at them. They're all templates.

**Live demo — the duplication problem:**

```cpp
int maxInt(int a, int b) { return (a > b) ? a : b; }
double maxDouble(double a, double b) { return (a > b) ? a : b; }
std::string maxString(std::string a, std::string b) { return (a > b) ? a : b; }
```

- Point out: identical logic, three separate functions to maintain. Add a fourth type, add a fourth function. This doesn't scale.

**Key line to say:**
> "You already know templates — you've been using them all along. `vector<Car>`, `unique_ptr<Engine>`, `map<string, int>` — that `<...>` syntax is you handing a type to a template someone else wrote. Today we write our own."

---

## 2. Function Templates (20 min)

**Talking points:**

- Syntax: `template <typename T>` immediately before a function declares "this function works for some type `T`, to be filled in later." `typename` (or the older, equivalent `class` keyword) just means "some type."
- The compiler figures out `T` from the arguments you pass — this is **template argument deduction**. You only need to specify it explicitly when the compiler can't figure it out on its own.

**Live demo:**

```cpp
template <typename T>
T maxVal(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << maxVal(3, 7) << std::endl;                 // T = int
    std::cout << maxVal(3.5, 2.1) << std::endl;              // T = double
    std::cout << maxVal(std::string("apple"), std::string("banana")) << std::endl; // T = std::string
}
```

- Run it. One function definition, three completely different generated versions under the hood.

**Ask:** "What happens if I call `maxVal(3, 3.5)` — an `int` and a `double`?" Let them guess, then show it: a compile error, because the compiler can't deduce a single `T` from two different argument types.

- Fix: `maxVal<double>(3, 3.5)` — explicitly telling the compiler what `T` is, so both arguments convert to it.

**Key line to say:**
> "`typename T` is a placeholder, not a real type — think of the angle brackets as a blank the compiler fills in, either by looking at your arguments or because you told it directly."

**Quick practice (5 min):**
> "Write a template function `swapVals(T& a, T& b)` that swaps two values of any type using a temporary variable. Test it with two `int`s and then two `std::string`s — same function, no changes needed."

---

## 3. Class Templates (20 min)

**Talking points:**

- The same idea applies to whole classes: `template <typename T>` before a class means every member — variables, method parameters, return types — can use `T` as a stand-in type, filled in when someone writes `MyClass<int>` or `MyClass<std::string>`.
- This is precisely what `std::vector<T>` is: a class template with one type parameter.

**Live demo — a generic `Stack<T>`:**

```cpp
#include <vector>
#include <string>

template <typename T>
class Stack {
private:
    std::vector<T> data;
public:
    void push(T value) { data.push_back(std::move(value)); }
    void pop() { data.pop_back(); }
    T& top() { return data.back(); }
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

int main() {
    Stack<int> ints;
    ints.push(1);
    ints.push(2);
    std::cout << ints.top() << std::endl; // 2

    Stack<std::string> words;
    words.push("hello");
    words.push("world");
    std::cout << words.top() << std::endl; // world
}
```

- Run it — same `Stack` class, two completely unrelated element types, zero duplicated code.

**Ask the audience:** "Notice `push` takes `T value` by value, then moves it into the vector — not `const T&`. Why might that matter if `T` is `std::unique_ptr<Engine>`?"

- Answer: a `unique_ptr` can't be copied, only moved. Taking by value plus `std::move` inside lets `Stack<T>` work for *both* ordinary copyable types **and** move-only types like `unique_ptr`, with one implementation. Taking `const T&` would silently break the moment someone tried `Stack<std::unique_ptr<Engine>>`.

**Key line to say:**
> "This is exactly why the STL containers are so careful about how they accept elements — now you've seen the actual mechanism, not just the behavior."

---

### ☕ Break (5 min)

---

## 5. Multiple Type Parameters & Non-Type Template Parameters (15 min)

**Talking points:**

- Templates aren't limited to one type parameter. `std::map<K, V>` takes two — that's a class template with two `typename` parameters.
- Template parameters also don't have to be *types*. A **non-type template parameter** is an actual value — usually an integer — that becomes part of the type itself, fixed at compile time. This is how `std::array<T, N>` gets a compile-time-fixed size without wasting any runtime memory tracking it.

**Live demo — two type parameters:**

```cpp
template <typename K, typename V>
class Pair {
public:
    K key;
    V value;
    Pair(K k, V v) : key(std::move(k)), value(std::move(v)) {}
    void print() const {
        std::cout << key << ": " << value << std::endl;
    }
};

int main() {
    Pair<std::string, int> p("red", 120);
    p.print(); // red: 120
}
```

- Point out: this is a simplified version of what `std::pair<K, V>` — and by extension every entry inside a `std::map<K, V>` — actually is.

**Live demo — a non-type parameter:**

```cpp
template <typename T, size_t N>
class FixedArray {
private:
    T data[N];
public:
    T& operator[](size_t i) { return data[i]; }
    size_t size() const { return N; }
};

int main() {
    FixedArray<double, 3> small;
    FixedArray<double, 100> big;
    std::cout << small.size() << " vs " << big.size() << std::endl; // 3 vs 100
}
```

**Key line to say:**
> "`FixedArray<double, 3>` and `FixedArray<double, 100>` are literally different types to the compiler — two entirely separate generated classes, sized at compile time, no dynamic resizing needed. That's the trade-off versus `vector`: fixed size, but zero heap allocation overhead."

**Quick practice (5 min):**
> "Before I run it — what do you think `FixedArray<int, 4> a; FixedArray<int, 5> b; a = b;` does when you try to compile it? (Answer: compile error — different `N` means genuinely different, incompatible types, even though both hold `int`.)"

---

## 6. Templates in the STL You Already Know (10 min)

**Talking points:**

- Time to connect the dots explicitly: `std::vector<T>`, `std::unique_ptr<T>`, `std::map<K,V>` are class templates, structured exactly like `Stack<T>` and `Pair<K,V>` above. `std::sort`, `std::find_if`, `std::max` are function templates, structured exactly like `maxVal<T>` above.
- One extra layer worth naming: STL *algorithms* like `sort` and `find_if` are templated not on the container type, but on the **iterator type**. That's why the same `std::find_if` call works whether you hand it iterators from a `vector`, a `set`, or a `map` — it never needs to know which container it came from, only how to step through a range.

**Key line to say:**
> "Every time you called `std::sort` or `std::find_if` last session, you were calling a function template. Nothing magic — exactly the mechanism you just wrote by hand, just written once by the people who built the standard library instead of by you."

---

## 7. Constraining Templates: `static_assert` and Concepts (10 min)

**Talking points:**

- A plain template accepts *any* type — including nonsensical ones. Instantiate `Stack<T>` with a type that doesn't support what you need, and you get a wall of confusing compiler error text pointing at internal implementation details, not your mistake.
- **`static_assert`** lets you write an explicit, compile-time check with a clear message, using `<type_traits>`:

  ```cpp
  #include <type_traits>

  template <typename T>
  T addNumbers(T a, T b) {
      static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
      return a + b;
  }
  ```

  Try `addNumbers(3, 4)` (fine) versus `addNumbers(std::string("a"), std::string("b"))` (fails immediately with your message, not an internal error dump).
- **C++20 concepts** are the newer, cleaner version of the same idea — a named, reusable requirement:

  ```cpp
  template <typename T>
  concept Numeric = std::is_arithmetic_v<T>;

  template <Numeric T>
  T addNumbers(T a, T b) {
      return a + b;
  }
  ```

  *(Skip live-running this if your compiler doesn't support C++20 — describe it verbally and move on.)*

**Key line to say:**
> "Concepts are basically named, reusable compile-time requirements — 'this type must support these operations' — and they turn a wall of template error text into one clear sentence pointing at the actual problem."

---

## 8. Wrap-up / Q&A (5 min)

**Recap in order:**

1. **Function templates** — write an algorithm once with `template <typename T>`; the compiler generates a version per type used, at zero runtime cost.
2. **Class templates** — the same idea for whole classes; this is literally what `vector<T>`, `unique_ptr<T>`, and `map<K,V>` are.
3. **Multiple & non-type template parameters** — templates can take more than one type (`Pair<K,V>`), and can take compile-time *values* as parameters (`FixedArray<T, N>`), which become part of the type itself.
4. **The STL is templates, all the way down** — containers are class templates, algorithms are function templates over iterator types.
5. **Constraining templates** — `static_assert` and, in C++20, `concepts` turn "any type compiles, badly, with a wall of errors" into "only sensible types compile, with a clear message when they don't."

**Tease next lesson:**
> "Next time: operator overloading and custom iterators — giving your own classes STL-like syntax (`container[i]`, `for (auto& x : container)`, `==`/`<` comparisons) — which is exactly what we'll need to build the `Enemy` system and turn-based combat for the dungeon crawler."

---

## 9. Project: Generalizing the Dungeon Crawler with a Template Container (25 min)

**The pitch:**
If you've been building the dungeon crawler, you may have noticed `Room` and `Player` each hand-rolled their own `vector<unique_ptr<Item>>` plus nearly identical add/find/remove logic. That duplication is precisely the problem templates exist to solve. Today we pull that logic out into one generic `OwningContainer<T>` class template, used by both — and because it's generic, it'll work unmodified for an `Enemy` type later, with no new container code needed.

**Rebuilt from scratch (self-contained — no prior file needed):**

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
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

// A generic owning container for any type with a getName() method.
template <typename T>
class OwningContainer {
private:
    vector<unique_ptr<T>> items;
public:
    void add(unique_ptr<T> item) {
        items.push_back(move(item));
    }

    unique_ptr<T> take(const string& itemName) {
        auto it = find_if(items.begin(), items.end(),
            [&](const unique_ptr<T>& i) { return i->getName() == itemName; });
        if (it == items.end()) return nullptr;
        unique_ptr<T> found = move(*it);
        items.erase(it);
        return found;
    }

    void list() const {
        for (const auto& item : items) {
            cout << " - " << item->getName() << endl;
        }
    }

    size_t size() const { return items.size(); }
};

class Room {
private:
    OwningContainer<Item> items;
public:
    void addItem(unique_ptr<Item> item) { items.add(move(item)); }
    unique_ptr<Item> takeItem(const string& name) { return items.take(name); }
};

class Player {
private:
    OwningContainer<Item> inventory;
public:
    void pickUp(unique_ptr<Item> item) {
        cout << "Picked up " << item->getName() << "." << endl;
        inventory.add(move(item));
    }
    void listInventory() const { inventory.list(); }
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

**Walk through it:**
> "`Room` and `Player` no longer each maintain their own `vector<unique_ptr<Item>>` plus their own copy of the add/find/remove logic — that logic now lives exactly once, inside `OwningContainer<T>`. Both classes just hold one and delegate to it. Notice `OwningContainer` doesn't know or care that `T` is `Item` specifically — the only thing it assumes is that `T` has a `getName()` method, because `take()` calls it. That's the whole point: swap in any type with a `getName()`, and this container works for it, unmodified."

**Suggested build order (spread across future sessions):**

1. Swap `Room` and `Player` over to `OwningContainer<Item>` — done above.
2. Add `Weapon` and `Armor` subclasses of `Item`.
3. Once you add an `Enemy` class with a `getName()` method, store enemies in a `Room` using `OwningContainer<Enemy>` — no new container code required. If it "just works," that's today's lesson landing.
4. Stretch goal: add a `static_assert` (or, on C++20, a `concept` named something like `Nameable`) constraining `T` to types with a `getName()` method, so instantiating `OwningContainer<int>` fails with one clear message instead of a wall of template errors from deep inside `take()`.
5. Next session's operator overloading will let `OwningContainer<T>` support `container[i]` and range-based `for` directly, instead of only `add`/`take`/`list`.

**Leave them with a challenge:**
> "Add an `Enemy` class with nothing but a `getName()` method, and store some in a `Room` using `OwningContainer<Enemy>` — reusing the exact same template, no new container code. Then, just to see it, try instantiating `OwningContainer<int>` and read the error message the compiler gives you. Think about how a `Nameable` concept could turn that into a single clear sentence, and we'll build one together next time we touch this project."

---

## Timing cheat sheet

| Section                                              | Minutes | Running total |
|-------------------------------------------------------|---------|----------------|
| Refresher: The Problem Templates Solve                | 10      | 10             |
| Function Templates (+ quick practice)                 | 20      | 30             |
| Class Templates                                       | 20      | 50             |
| Break                                                  | 5       | 55             |
| Multiple & Non-Type Template Parameters (+ practice)   | 15      | 70             |
| Templates in the STL You Already Know                  | 10      | 80             |
| Constraining Templates (`static_assert` / concepts)    | 10      | 90             |
| Wrap-up / Q&A                                          | 5       | 95             |
| Project: generalizing the dungeon crawler with templates | 25    | 120            |

**Running long?** Cut points, in order of safety:

1. Trim the `FixedArray<T, N>` non-type parameter demo to a verbal description only; keep `Pair<K,V>`.
2. Reduce Section 7 to the `static_assert` example only — mention concepts verbally, don't live-code them.
3. Shorten the project section to the pitch and the code walkthrough — skip live-coding it and share the block as a handout.

**Running short / audience is fast?**

- Before running the `Stack<T>` demo, have them predict what happens if `push` took `const T&` instead of `T` — then show them the compile error with `Stack<unique_ptr<Engine>>`.
- Ask them to write a `template <typename T> bool allEqual(T a, T b, T c)` that returns whether all three are equal — quick extra reps with deduction before moving on.
- Have them attempt the `Nameable` concept from the project's stretch goal live, instead of leaving it as homework.
