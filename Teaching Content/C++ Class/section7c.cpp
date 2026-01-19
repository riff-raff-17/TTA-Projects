// 07b_vector_operations.cpp
// Section 7: Common std::vector Operations and Uses

#include <algorithm> // sort, reverse, find, count, remove
#include <iostream>
#include <vector>
#include <string>
using namespace std;

// Prints a vector neatly
void printVector(const vector<int>& v, const string& label) {
    cout << label << " (size=" << v.size() << ", capacity=" << v.capacity() << "): ";
    for (int x : v) cout << x << " ";
    cout << endl;
}

// Adds a value to every element (demonstrates passing by reference)
void addToAll(vector<int>& v, int add) {
    for (int& x : v) x += add;
}

// Sums elements (demonstrates passing by const reference)
int sumVector(const vector<int>& v) {
    int total = 0;
    for (int x : v) total += x;
    return total;
}

int main() {
    cout << "=== 1) Creating and initializing vectors ===\n";
    vector<int> a; // empty
    vector<int> b(5); // 5 zeros: 0 0 0 0 0
    vector<int> c(4, 7); // 4 copies of 7: 7 7 7 7
    vector<int> d = {3, 1, 4, 1, 5};

    printVector(a, "a");
    printVector(b, "b");
    printVector(c, "c");
    printVector(d, "d");

    cout << "\n=== 2) size vs capacity, reserve, shrink_to_fit ===\n";
    vector<int> v;
    printVector(v, "v (start)");

    v.reserve(10); // pre-allocate space to reduce re-allocations
    printVector(v, "v after reserve(10)");

    for (int i = 1; i <= 6; i++) v.push_back(i * 10);
    printVector(v, "v after push_back 6 values");

    // shrink_to_fit requests to reduce capacity to size (not always guaranteed, but commonly works)
    v.shrink_to_fit();
    printVector(v, "v after shrink_to_fit()");

    cout << "\n=== 3) Adding/removing: push_back, pop_back, clear ===\n";
    v.push_back(999);
    printVector(v, "v after push_back(999)");

    v.pop_back(); // removes last element
    printVector(v, "v after pop_back()");

    vector<int> temp = {1, 2, 3};
    printVector(temp, "temp");
    temp.clear(); // removes all elements (size becomes 0)
    printVector(temp, "temp after clear()");

    cout << "\n=== 4) Accessing elements: [], at(), front(), back() ===\n";
    // [] does NOT do bounds checking
    cout << "v[0] = " << v[0] << "\n";

    // at() DOES bounds checking and can throw an exception if out of range
    cout << "v.at(1) = " << v.at(1) << "\n";

    cout << "front() = " << v.front() << "\n";
    cout << "back()  = " << v.back() << "\n";

    cout << "\n=== 5) Iteration styles ===\n";
    cout << "Index-based: ";
    for (int i = 0; i < (int)v.size(); i++) cout << v[i] << " ";
    cout << "\n";

    cout << "Range-based for: ";
    for (int x : v) cout << x << " ";
    cout << "\n";

    cout << "Using iterators: ";
    for (auto it = v.begin(); it != v.end(); ++it) cout << *it << " ";
    cout << "\n";

    cout << "\n=== 6) insert and erase ===\n";
    vector<int> e = {10, 20, 30, 40};
    printVector(e, "e");

    // insert value at position (begin()+index)
    e.insert(e.begin() + 2, 999); // before 30
    printVector(e, "e after insert at index 2");

    // erase one element at index 1
    e.erase(e.begin() + 1); // removes 20
    printVector(e, "e after erase index 1");

    // erase a range: remove elements [index 1, index 3) (end index not included)
    // (Make sure indices are valid before doing this in real programs.)
    if (e.size() >= 3) {
        e.erase(e.begin() + 1, e.begin() + 3);
        printVector(e, "e after erase range [1,3)");
    }

    cout << "\n=== 7) Removing all occurrences of a value (erase-remove idiom) ===\n";
    vector<int> r = {1, 2, 2, 3, 2, 4};
    printVector(r, "r");

    // remove moves "kept" elements forward and returns new logical end
    r.erase(remove(r.begin(), r.end(), 2), r.end());
    printVector(r, "r after removing all 2s");

    cout << "\n=== 8) Searching and counting ===\n";
    vector<int> s = {5, 1, 5, 2, 5, 3};
    printVector(s, "s");

    int target = 2;
    auto it = find(s.begin(), s.end(), target);
    if (it != s.end()) {
        int index = (int)distance(s.begin(), it);
        cout << "Found " << target << " at index " << index << "\n";
    } else {
        cout << target << " not found\n";
    }

    cout << "Count of 5 = " << count(s.begin(), s.end(), 5) << "\n";

    cout << "\n=== 9) Sorting and reversing ===\n";
    vector<int> t = {9, 3, 7, 1, 4};
    printVector(t, "t");

    sort(t.begin(), t.end()); // ascending
    printVector(t, "t after sort ascending");

    reverse(t.begin(), t.end());
    printVector(t, "t after reverse (now descending)");

    cout << "\n=== 10) 2D vectors (matrix / grid) ===\n";
    int rows = 3, cols = 4;
    vector<vector<int>> grid(rows, vector<int>(cols, 0)); // 3x4 filled with 0s

    // Set a few values
    grid[0][1] = 5;
    grid[2][3] = 9;

    cout << "grid:\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            cout << grid[r][c] << " ";
        }
        cout << "\n";
    }

    cout << "\n=== 11) Passing vectors to functions ===\n";
    vector<int> p = {10, 20, 30};
    printVector(p, "p before addToAll");

    addToAll(p, 7);
    printVector(p, "p after addToAll(+7)");

    cout << "sumVector(p) = " << sumVector(p) << "\n";

    cout << "\nDone.\n";
    return 0;
}
