// =====================================================================
// templates_and_generics.cpp
// Companion code file for the "Templates & Generic Programming" session.
// Read this alongside templates_generic_programming_session.md — that
// file has the full talking points, this file is just the runnable code
// with comments explaining WHY each piece is written the way it is.
//
// Compile with (C++17 minimum; C++20 optional, only affects Section 7):
//   g++ -std=c++17 templates_and_generics.cpp -o templates_demo
//   ./templates_demo
//
// Structure: one function per lesson section, all called in order from
// main() at the very bottom. Comment out a call in main() if you want
// to skip a section while live-coding or reviewing.
// =====================================================================

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <algorithm>
#include <type_traits>   // needed for static_assert(std::is_arithmetic<T>...)

using namespace std;


// =====================================================================
// SECTION 1 — The problem templates solve (reference only, not called)
// =====================================================================
// Before templates, you'd hand-write a near-identical function per type.
// This is the "bad" version we're trying to avoid — left here as a
// comment for reference, not compiled or called anywhere below.
//
// int maxInt(int a, int b) { return (a > b) ? a : b; }
// double maxDouble(double a, double b) { return (a > b) ? a : b; }
// string maxString(string a, string b) { return (a > b) ? a : b; }
//
// Three functions, identical logic, three places to maintain a bug fix.
// Section 2 replaces all three with ONE function template.


// =====================================================================
// SECTION 2 — Function templates
// =====================================================================

// `template <typename T>` means: "T is a placeholder type, to be filled
// in later — either by the compiler (deduced from the arguments you
// pass) or explicitly by us, like maxVal<double>(...)."
template <typename T>
T maxVal(T a, T b) {
    return (a > b) ? a : b;
}

// Quick-practice answer from the lesson: a generic swap using a temp
// variable. Works for any type T — try it with int, then with string.
template <typename T>
void swapVals(T& a, T& b) {
    T temp = a;   // hang on to a's original value before overwriting it
    a = b;
    b = temp;
}

void section2_functionTemplates() {
    cout << "\n=== Section 2: Function Templates ===\n";

    // Compiler deduces T = int from the argument types — no <int> needed.
    cout << "maxVal(3, 7)           = " << maxVal(3, 7) << endl;

    // Compiler deduces T = double.
    cout << "maxVal(3.5, 2.1)       = " << maxVal(3.5, 2.1) << endl;

    // Compiler deduces T = string.
    cout << "maxVal(apple, banana)  = "
         << maxVal(string("apple"), string("banana")) << endl;

    // Mixed types (int, double) — the compiler can't deduce ONE T from
    // two different argument types. Uncommenting the next line is a
    // compile error — try it and read the message:
    // cout << maxVal(3, 3.5) << endl;

    // Fix: tell the compiler explicitly what T is, so both arguments
    // convert to it before the call happens.
    cout << "maxVal<double>(3, 3.5) = " << maxVal<double>(3, 3.5) << endl;

    // swapVals in action — same function, two unrelated types.
    int x = 1, y = 2;
    swapVals(x, y);
    cout << "swapVals(int): x=" << x << " y=" << y << endl;   // x=2 y=1

    string s1 = "left", s2 = "right";
    swapVals(s1, s2);
    cout << "swapVals(string): s1=" << s1 << " s2=" << s2 << endl;
}


// =====================================================================
// SECTION 3 — Class templates
// =====================================================================

// A generic Stack, backed internally by a vector<T>. Works for ANY type
// T — the compiler generates a fresh, separate class each time you
// write Stack<SomeType>.
template <typename T>
class Stack {
private:
    vector<T> data;

public:
    // Takes T BY VALUE (not const T&), then moves it into the vector.
    // Why this matters: it lets push() work for BOTH ordinary copyable
    // types (int, string) AND move-only types (unique_ptr<...>).
    // A const T& parameter would compile fine for int/string but break
    // the moment someone wrote Stack<unique_ptr<Engine>>, since a
    // unique_ptr can be moved but never copied.
    void push(T value) {
        data.push_back(move(value));
    }

    void pop() {
        data.pop_back();
    }

    T& top() {
        return data.back();
    }

    bool empty() const {
        return data.empty();
    }

    size_t size() const {
        return data.size();
    }
};

void section3_classTemplates() {
    cout << "\n=== Section 3: Class Templates ===\n";

    Stack<int> ints;
    ints.push(1);
    ints.push(2);
    cout << "ints.top()  = " << ints.top() << endl;   // 2

    Stack<string> words;
    words.push("hello");
    words.push("world");
    cout << "words.top() = " << words.top() << endl;  // world

    // Proof this also works for a move-only type: uncomment to try.
    // Stack<unique_ptr<int>> ptrs;
    // ptrs.push(make_unique<int>(42));   // fine — push takes by value + moves
    // cout << *ptrs.top() << endl;
}


// =====================================================================
// SECTION 5 — Multiple type parameters & non-type template parameters
// =====================================================================

// Two type parameters — a simplified version of what std::pair<K, V>
// (and every entry inside a std::map<K, V>) actually is under the hood.
template <typename K, typename V>
class Pair {
public:
    K key;
    V value;

    Pair(K k, V v) : key(move(k)), value(move(v)) {}

    void print() const {
        cout << key << ": " << value << endl;
    }
};

// A NON-TYPE template parameter: N is a VALUE (a size_t), not a type —
// but it becomes part of the type itself. FixedArray<double,3> and
// FixedArray<double,100> are two entirely different generated classes.
template <typename T, size_t N>
class FixedArray {
private:
    T data[N];   // size fixed at COMPILE time — no heap allocation at all

public:
    T& operator[](size_t i) {
        return data[i];
    }

    size_t size() const {
        return N;
    }
};

void section5_multipleAndNonTypeParams() {
    cout << "\n=== Section 5: Multiple & Non-Type Template Parameters ===\n";

    Pair<string, int> p("red", 120);
    p.print();   // red: 120

    FixedArray<double, 3> small;
    FixedArray<double, 100> big;
    cout << "small.size() = " << small.size()
         << ", big.size() = " << big.size() << endl;   // 3 vs 100

    // small = big;
    // ^ compile error if uncommented: FixedArray<double,3> and
    //   FixedArray<double,100> are DIFFERENT TYPES, even though both
    //   hold doubles. The size N is part of the type, not just a field.
}


// =====================================================================
// SECTION 6 — Templates in the STL you already know
// =====================================================================
// No real new syntax here — this section is about recognizing that
// std::vector<T>, std::unique_ptr<T>, std::map<K,V> are class templates
// structured exactly like Stack<T> and Pair<K,V> above, and that
// std::sort / std::find_if are function templates structured exactly
// like maxVal<T>. The demo below is a hand-written find_if, side by
// side with the real one, to prove there's no hidden magic.

template <typename Iter, typename Predicate>
Iter myFindIf(Iter begin, Iter end, Predicate pred) {
    for (Iter it = begin; it != end; ++it) {
        if (pred(*it)) {
            return it;
        }
    }
    return end;
}

void section6_stlIsTemplates() {
    cout << "\n=== Section 6: Templates in the STL ===\n";

    vector<int> speeds = {50, 120, 30, 95};

    // Our hand-written version...
    auto it1 = myFindIf(speeds.begin(), speeds.end(),
                         [](int s) { return s > 100; });

    // ...versus the real std::find_if. Same idea, same result.
    auto it2 = find_if(speeds.begin(), speeds.end(),
                        [](int s) { return s > 100; });

    if (it1 != speeds.end()) cout << "myFindIf found:     " << *it1 << endl;
    if (it2 != speeds.end()) cout << "std::find_if found: " << *it2 << endl;
}


// =====================================================================
// SECTION 7 — Constraining templates: static_assert and concepts
// =====================================================================

// static_assert runs at COMPILE time, before the program ever executes.
// If T isn't arithmetic (int, double, etc.), this fails immediately
// with OUR message, instead of a wall of confusing errors from deep
// inside the function body.
template <typename T>
T addNumbers(T a, T b) {
    static_assert(std::is_arithmetic<T>::value, "T must be a numeric type");
    return a + b;
}

// C++20 concepts: a named, reusable version of the same idea.
// Guarded with a feature-test macro so this file still compiles fine
// under C++17 — if your compiler doesn't define __cpp_concepts, this
// whole block is simply skipped, and section7 below notices and adapts.
#if defined(__cpp_concepts)
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

template <Numeric T>
T addNumbersConcept(T a, T b) {
    return a + b;
}
#endif

void section7_constrainingTemplates() {
    cout << "\n=== Section 7: Constraining Templates ===\n";

    cout << "addNumbers(3, 4) = " << addNumbers(3, 4) << endl;

    // Uncommenting this line fails to compile — with our static_assert
    // message, not a wall of unrelated template errors:
    // addNumbers(string("a"), string("b"));

#if defined(__cpp_concepts)
    cout << "addNumbersConcept(3, 4) = " << addNumbersConcept(3, 4) << endl;
    // addNumbersConcept(string("a"), string("b")); // also fails to compile
#else
    cout << "(Concepts demo skipped — compiler predates C++20 concepts.)" << endl;
#endif
}


// =====================================================================
// SECTION 9 — Project: generalizing the dungeon crawler with templates
// =====================================================================

// Item hierarchy — same shape as the previous session's project.
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
    void use() override {
        cout << "Drank " << name << ". Feeling better!" << endl;
    }
};

// A generic owning container for ANY type T that has a getName() method.
// This replaces what used to be two near-identical, hand-rolled
// vector<unique_ptr<Item>> implementations — one inside Room, one
// inside Player. Now that logic lives exactly once.
template <typename T>
class OwningContainer {
private:
    vector<unique_ptr<T>> items;

public:
    void add(unique_ptr<T> item) {
        items.push_back(move(item));
    }

    // Finds, removes, and returns the named item — or nullptr if absent.
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

    size_t size() const {
        return items.size();
    }
};

class Room {
private:
    OwningContainer<Item> items;

public:
    void addItem(unique_ptr<Item> item) {
        items.add(move(item));
    }

    unique_ptr<Item> takeItem(const string& name) {
        return items.take(name);
    }
};

class Player {
private:
    OwningContainer<Item> inventory;

public:
    void pickUp(unique_ptr<Item> item) {
        cout << "Picked up " << item->getName() << "." << endl;
        inventory.add(move(item));
    }

    void listInventory() const {
        inventory.list();
    }
};

void section9_projectDemo() {
    cout << "\n=== Section 9: Project — OwningContainer<T> ===\n";

    Room startRoom;
    startRoom.addItem(make_unique<Potion>("Health Potion"));

    Player player;
    unique_ptr<Item> picked = startRoom.takeItem("Health Potion");
    if (picked) {
        player.pickUp(move(picked));
    }

    cout << "Player inventory:" << endl;
    player.listInventory();

    // Challenge from the lesson: add an Enemy class with nothing but a
    // getName() method, then try:
    //     OwningContainer<Enemy> enemies;
    // ...with ZERO new container code written. If it compiles and
    // works, the point of this whole session has landed.
}


// =====================================================================
// main — runs every section's demo in order.
// Comment out a call below to skip that section while reviewing.
// =====================================================================
int main() {
    section2_functionTemplates();
    section3_classTemplates();
    section5_multipleAndNonTypeParams();
    section6_stlIsTemplates();
    section7_constrainingTemplates();
    section9_projectDemo();
    return 0;
}
