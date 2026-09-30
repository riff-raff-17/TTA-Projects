// stl_containers_intro.cpp
//
// A short, self-contained intro to the four core STL containers:
//   vector, map, unordered_map, set
// plus a quick tour of the core STL algorithms:
//   sort, find_if, count_if, transform, accumulate
//
// Compile:  g++ -std=c++17 stl_containers_intro.cpp -o containers
// Run:      ./containers

#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>

using namespace std;

int main() {
    // -----------------------------------------------------------
    // 1. std::vector<T> — resizable array, contiguous memory,
    //    fast random access. Your default "list of things."
    // -----------------------------------------------------------
    cout << "--- vector ---" << endl;

    vector<int> speeds = {50, 120, 30, 95};
    speeds.push_back(75); // grows automatically

    for (int s : speeds) {
        cout << s << " ";
    }
    cout << endl;

    cout << "First speed: " << speeds[0] << endl;
    cout << "Total speeds: " << speeds.size() << endl;

    // -----------------------------------------------------------
    // 2. std::map<K, V> — sorted key/value table (red-black tree).
    //    Keys always come out in order when you iterate.
    // -----------------------------------------------------------
    cout << "\n--- map ---" << endl;

    map<string, int> speedByColor;
    speedByColor["red"] = 120;
    speedByColor["blue"] = 95;
    speedByColor["green"] = 30;

    // iterates in sorted key order: blue, green, red
    for (const auto& [color, speed] : speedByColor) {
        cout << color << ": " << speed << endl;
    }

    // -----------------------------------------------------------
    // 3. std::unordered_map<K, V> — same idea as map, but a hash
    //    table instead of a tree. Faster lookups, no guaranteed order.
    // -----------------------------------------------------------
    cout << "\n--- unordered_map ---" << endl;

    unordered_map<string, int> fastLookup;
    fastLookup["red"] = 120;
    fastLookup["blue"] = 95;

    cout << "Red car speed: " << fastLookup["red"] << endl;
    // order not guaranteed if you loop over this one — don't rely on it

    // -----------------------------------------------------------
    // 4. std::set<T> — like a map with no value, just unique keys
    //    in sorted order. Good for "have I seen this before?"
    // -----------------------------------------------------------
    cout << "\n--- set ---" << endl;

    set<string> visitedRooms;
    visitedRooms.insert("Entrance");
    visitedRooms.insert("Hallway");
    visitedRooms.insert("Entrance"); // duplicate: silently ignored

    cout << "Rooms visited: " << visitedRooms.size() << endl;
    for (const auto& room : visitedRooms) {
        cout << " - " << room << endl;
    }

    // -----------------------------------------------------------
    // 5. STL Algorithms — <algorithm> and <numeric> replace loops
    //    you'd otherwise write by hand. Most take a begin/end
    //    iterator pair, and often a lambda describing what to do.
    // -----------------------------------------------------------
    cout << "\n--- algorithms ---" << endl;

    vector<int> algoSpeeds = {50, 120, 30, 95};

    // sort — rearranges elements in place, ascending by default
    sort(algoSpeeds.begin(), algoSpeeds.end());
    cout << "Sorted: ";
    for (int s : algoSpeeds) cout << s << " ";
    cout << endl;

    // find_if — first element matching a condition (a lambda here)
    auto fast = find_if(algoSpeeds.begin(), algoSpeeds.end(),
        [](int s) { return s > 100; });
    if (fast != algoSpeeds.end()) {
        cout << "First speed over 100: " << *fast << endl;
    }

    // count_if — how many elements match a condition
    int slowCount = count_if(algoSpeeds.begin(), algoSpeeds.end(),
        [](int s) { return s < 60; });
    cout << "Speeds under 60: " << slowCount << endl;

    // transform — build a new sequence by applying a function to each element
    vector<int> doubled(algoSpeeds.size());
    transform(algoSpeeds.begin(), algoSpeeds.end(), doubled.begin(),
        [](int s) { return s * 2; });
    cout << "Doubled: ";
    for (int s : doubled) cout << s << " ";
    cout << endl;

    // accumulate — reduce a range down to a single value (sum by default)
    int total = accumulate(algoSpeeds.begin(), algoSpeeds.end(), 0);
    cout << "Total of all speeds: " << total << endl;

    return 0;
}
