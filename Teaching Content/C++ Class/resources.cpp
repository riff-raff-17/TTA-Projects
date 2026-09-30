// resources.cpp
//
// Each section below matches a section of the lesson markdown, in order.
// main() at the bottom calls every demo one after another with a printed
// header, so it compiles and runs as-is. If you're building this up live
// with a class, comment out the later demo calls in main() and uncomment
// them one at a time as you reach that section of the lesson.
//

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>

using namespace std;

void section(const string& title) {
    cout << "\n=== " << title << " ===" << endl;
}

// ---------------------------------------------------------------------
// Section 1: Refresher: Pointers, new, and Heap vs. Stack
// ---------------------------------------------------------------------

class Engine {
public:
    Engine() { cout << "Engine built." << endl; }
    ~Engine() { cout << "Engine destroyed." << endl; }
    void start() { cout << "Vroom." << endl; }
};

void pointerRefresherDemo() {
    Engine* e = new Engine();  // "Engine built." prints; e now holds an address
    e->start();                // follow the address, call start(); "Vroom." prints
    delete e;                  // clean up manually here so this demo alone doesn't leak
}

// ---------------------------------------------------------------------
// Section 2: Why RAII?
// ---------------------------------------------------------------------

void leaky() {
    Engine* e = new Engine();
    e->start();
    // no delete e; -> "Engine destroyed." never prints
}

void leakyWithException(bool badInput) {
    Engine* e = new Engine();
    if (badInput) {
        throw runtime_error("bad input");  // jumps out immediately —
        // delete e; below never runs, even though it's right there in the code
    }
    e->start();
    delete e;
}

void leakyWithExceptionDemo() {
    try {
        leakyWithException(true);
    } catch (const exception& ex) {
        cout << "Caught exception: " << ex.what() << endl;
        cout << "Notice: no \"Engine destroyed.\" printed above. That Engine leaked." << endl;
    }
}

// ---------------------------------------------------------------------
// Section 3: Smart Pointers
// ---------------------------------------------------------------------

void notLeakyAnymore() {
    unique_ptr<Engine> e = make_unique<Engine>();
    e->start();
    // no delete needed — destructor runs automatically here
}

void safeWithException(bool badInput) {
    unique_ptr<Engine> e = make_unique<Engine>();
    if (badInput) {
        throw runtime_error("bad input");
        // no delete needed — e's destructor runs during unwinding anyway
    }
    e->start();
}

void safeWithExceptionDemo() {
    try {
        safeWithException(true);
    } catch (const exception& ex) {
        cout << "Caught exception: " << ex.what() << endl;
        cout << "Notice: \"Engine destroyed.\" DID print above, even though we threw early." << endl;
    }
}

void uniquePtrMoveDemo() {
    unique_ptr<Engine> e = make_unique<Engine>();
    unique_ptr<Engine> e2 = move(e); // ownership transfers
    if (!e) {
        cout << "e is now empty after the move." << endl;
    }
    e2->start();

    // Uncomment the next line to see the compile error from trying to copy a unique_ptr:
    // unique_ptr<Engine> e3 = e2;
}

void sharedPtrDemo() {
    shared_ptr<Engine> e1 = make_shared<Engine>();
    cout << "Owners: " << e1.use_count() << endl; // 1
    {
        shared_ptr<Engine> e2 = e1; // copy is allowed
        cout << "Owners: " << e1.use_count() << endl; // 2
    } // e2 destroyed, but Engine survives — e1 still owns it
    cout << "Owners: " << e1.use_count() << endl; // 1
} // now the Engine is destroyed

// weak_ptr: demonstrate the reference-cycle problem, then the fix.
// (Named CycleRoom/CycleDoor here to avoid clashing with the dungeon-crawler
// Room class defined later in this file.)

struct CycleRoom;

struct CycleDoorLeaky {
    shared_ptr<CycleRoom> connectsTo;
    ~CycleDoorLeaky() { cout << "CycleDoorLeaky destroyed." << endl; }
};

struct CycleRoom {
    shared_ptr<CycleDoorLeaky> door;
    ~CycleRoom() { cout << "CycleRoom destroyed." << endl; }
};

void cycleLeakDemo() {
    auto room = make_shared<CycleRoom>();
    auto door = make_shared<CycleDoorLeaky>();
    room->door = door;
    door->connectsTo = room;
    cout << "Function ending — watch: neither destructor will print." << endl;
    // neither destructor prints when this function ends —
    // room and door each keep the other's count above zero forever
}

struct CycleRoomFixed;

struct CycleDoorFixed {
    weak_ptr<CycleRoomFixed> connectsTo; // no longer keeps the room alive
    ~CycleDoorFixed() { cout << "CycleDoorFixed destroyed." << endl; }
};

struct CycleRoomFixed {
    shared_ptr<CycleDoorFixed> door;
    ~CycleRoomFixed() { cout << "CycleRoomFixed destroyed." << endl; }
};

void weakPtrFixDemo() {
    auto room = make_shared<CycleRoomFixed>();
    auto door = make_shared<CycleDoorFixed>();
    room->door = door;
    door->connectsTo = room;

    if (auto lockedRoom = door->connectsTo.lock()) {
        cout << "Room is still alive, accessed safely via lock()." << endl;
    }

    cout << "Function ending — both destructors should print now." << endl;
}

// Tie smart pointers back to a Car that owns an Engine.
class Car {
private:
    unique_ptr<Engine> engine;
    int speed = 0;
public:
    Car() : engine(make_unique<Engine>()) {
        cout << "Car built with its own engine." << endl;
    }
    void start() { engine->start(); }
    void accelerate() { speed += 10; }
    int getSpeed() const { return speed; }
};

void carOwnsEngineDemo() {
    Car car;
    car.start();
    // when `car` goes out of scope, its unique_ptr<Engine> destroys
    // the Engine automatically — no destructor code needed on our part
}

// ---------------------------------------------------------------------
// Section 4: Move Semantics
// ---------------------------------------------------------------------

void moveSemanticsDemo() {
    unique_ptr<Engine> a = make_unique<Engine>();
    unique_ptr<Engine> b = move(a);

    if (!a) {
        cout << "a is now empty." << endl;
    }
    b->start(); // b owns the Engine now
}

void moveIntoVectorDemo() {
    vector<unique_ptr<Engine>> engines;
    unique_ptr<Engine> e = make_unique<Engine>();
    engines.push_back(move(e)); // must move — can't copy a unique_ptr
    // e is now empty; the vector owns the Engine
    cout << "Vector now owns " << engines.size() << " engine(s)." << endl;
}

// ---------------------------------------------------------------------
// Section 5: STL — Containers
// ---------------------------------------------------------------------

void garage() {
    vector<unique_ptr<Car>> cars;
    cars.push_back(make_unique<Car>());
    cars.push_back(make_unique<Car>());

    for (const auto& car : cars) {
        car->start();
    }
    // when `cars` goes out of scope, every Car AND every Engine
    // inside it is destroyed automatically. No leaks, no manual cleanup.
}

void mapDemo() {
    map<string, int> speedByColor;
    speedByColor["red"] = 120;
    speedByColor["blue"] = 95;
    cout << "Red car speed: " << speedByColor["red"] << endl;

    // iterate in sorted key order:
    for (const auto& [color, speed] : speedByColor) {
        cout << color << ": " << speed << endl;
    }
}

void unorderedMapAndSetDemo() {
    unordered_map<string, int> fastLookup;
    fastLookup["red"] = 120; // same interface as map, different internals
    cout << "Fast lookup for red: " << fastLookup["red"] << endl;

    set<string> visitedRooms;
    visitedRooms.insert("Entrance");
    visitedRooms.insert("Entrance"); // no-op, already present
    cout << "Rooms visited: " << visitedRooms.size() << endl; // 1
}

// ---------------------------------------------------------------------
// Section 6: STL Algorithms
// ---------------------------------------------------------------------

void sortAndFindDemo() {
    vector<int> speeds = {50, 120, 30, 95};
    sort(speeds.begin(), speeds.end());
    // speeds is now {30, 50, 95, 120}

    cout << "Sorted speeds: ";
    for (int s : speeds) cout << s << " ";
    cout << endl;

    auto fast = find_if(speeds.begin(), speeds.end(),
        [](int s) { return s > 100; });
    if (fast != speeds.end()) {
        cout << "First speed over 100: " << *fast << endl;
    }
}

void countTransformAccumulateDemo() {
    vector<int> speeds = {50, 120, 30, 95};

    int fastCount = count_if(speeds.begin(), speeds.end(),
        [](int s) { return s > 60; });
    cout << "Cars faster than 60: " << fastCount << endl;

    vector<int> doubled(speeds.size());
    transform(speeds.begin(), speeds.end(), doubled.begin(),
        [](int s) { return s * 2; });
    cout << "Doubled speeds: ";
    for (int s : doubled) cout << s << " ";
    cout << endl;

    int total = accumulate(speeds.begin(), speeds.end(), 0);
    cout << "Total speed: " << total << endl;
}

void sortUniquePtrVectorDemo() {
    vector<unique_ptr<Car>> cars;
    cars.push_back(make_unique<Car>());
    cars.push_back(make_unique<Car>());
    cars.push_back(make_unique<Car>());

    cars[0]->accelerate(); // speed 10
    cars[1]->accelerate();
    cars[1]->accelerate(); // speed 20
    // cars[2] stays at speed 0

    sort(cars.begin(), cars.end(),
        [](const unique_ptr<Car>& a, const unique_ptr<Car>& b) {
            return a->getSpeed() < b->getSpeed();
        });

    cout << "Cars sorted by speed: ";
    for (const auto& car : cars) cout << car->getSpeed() << " ";
    cout << endl;
}

// ---------------------------------------------------------------------
// Section 8: The Project — Dungeon Crawler starter skeleton
// ---------------------------------------------------------------------

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

void dungeonCrawlerSkeletonDemo() {
    Room startRoom;
    startRoom.addItem(make_unique<Potion>("Health Potion"));

    Player player;
    unique_ptr<Item> picked = startRoom.takeItem("Health Potion");
    if (picked) {
        player.pickUp(move(picked));
    }

    player.listInventory();
}

// ---------------------------------------------------------------------
// main: runs every demo in lesson order
// ---------------------------------------------------------------------

int main() {
    section("1. Pointer refresher");
    pointerRefresherDemo();

    section("2. Why RAII? -- leaky()");
    leaky();

    section("2. Why RAII? -- leakyWithException()");
    leakyWithExceptionDemo();

    section("3. Smart Pointers -- unique_ptr, no leak");
    notLeakyAnymore();

    section("3. Smart Pointers -- unique_ptr survives an exception");
    safeWithExceptionDemo();

    section("3. Smart Pointers -- unique_ptr move");
    uniquePtrMoveDemo();

    section("3. Smart Pointers -- shared_ptr use_count()");
    sharedPtrDemo();

    section("3. Smart Pointers -- shared_ptr reference cycle (leaks on purpose)");
    cycleLeakDemo();

    section("3. Smart Pointers -- weak_ptr fixes the cycle");
    weakPtrFixDemo();

    section("3. Smart Pointers -- Car owns an Engine");
    carOwnsEngineDemo();

    section("4. Move Semantics -- basic move");
    moveSemanticsDemo();

    section("4. Move Semantics -- move into a vector");
    moveIntoVectorDemo();

    section("5. STL Containers -- vector of unique_ptr<Car>");
    garage();

    section("5. STL Containers -- map");
    mapDemo();

    section("5. STL Containers -- unordered_map and set");
    unorderedMapAndSetDemo();

    section("6. STL Algorithms -- sort and find_if");
    sortAndFindDemo();

    section("6. STL Algorithms -- count_if, transform, accumulate");
    countTransformAccumulateDemo();

    section("6. STL Algorithms -- sorting a vector<unique_ptr<Car>>");
    sortUniquePtrVectorDemo();

    section("8. Project -- dungeon crawler starter skeleton");
    dungeonCrawlerSkeletonDemo();

    return 0;
}
