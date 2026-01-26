// 06b_passing_to_functions.cpp
// Section 6b: Ways to Pass Arguments to Functions in C++

#include <iostream>
#include <string>
#include <vector>
using namespace std;

// 1) Pass by VALUE: makes a copy
void tryToChangeValue(int x) {
    x = 999; // only changes the local copy
}

vector<int> addOneToEach_ByValue(vector<int> v) {
    // v is a copy; caller's vector will not change
    for (int& x : v) x += 1;
    return v; // returns the modified copy
}

// 2) Pass by REFERENCE: no copy, can modify original
void changeByRef(int& x) {
    x = 999; // changes the caller's variable
}

void addOneToEach_ByRef(vector<int>& v) {
    for (int& x : v) x += 1; // modifies caller's vector
}

// 3) Pass by CONST REFERENCE: no copy, read-only
int sumVector_ConstRef(const vector<int>& v) {
    int total = 0;
    for (int x : v) total += x;
    // v.push_back(123); // would be a compiler error (read-only)
    return total;
}

void printVector_ConstRef(const vector<int>& v, const string& label) {
    cout << label << " (size=" << v.size() << "): ";
    for (int x : v) cout << x << " ";
    cout << "\n";
}

// 4) Pass by POINTER: can be null; can modify via *ptr
void changeByPointer(int* p) {
    if (p != nullptr) {
        *p = 777; // modifies caller's variable through pointer
    }
}

// Optional / maybe-present parameter style
void printIfNotNull(const string* s) {
    if (s) cout << "String: " << *s << "\n";
    else   cout << "String: (null)\n";
}

int main() {
    cout << "=== 1) Pass by value (copy) ===\n";
    int a = 10;
    cout << "Before tryToChangeValue: a = " << a << "\n";
    tryToChangeValue(a);
    cout << "After tryToChangeValue:  a = " << a << " (unchanged)\n\n";

    vector<int> v1 = {1, 2, 3};
    printVector_ConstRef(v1, "v1 before addOneToEach_ByValue");

    vector<int> v1_modified = addOneToEach_ByValue(v1);
    printVector_ConstRef(v1, "v1 after addOneToEach_ByValue (original)");
    printVector_ConstRef(v1_modified, "returned copy (modified)");
    cout << "\n";

    cout << "=== 2) Pass by reference (&) ===\n";
    int b = 20;
    cout << "Before changeByRef: b = " << b << "\n";
    changeByRef(b);
    cout << "After changeByRef:  b = " << b << " (changed)\n\n";

    vector<int> v2 = {1, 2, 3};
    printVector_ConstRef(v2, "v2 before addOneToEach_ByRef");
    addOneToEach_ByRef(v2);
    printVector_ConstRef(v2, "v2 after addOneToEach_ByRef (changed)");
    cout << "\n";

    cout << "=== 3) Pass by const reference (const &) ===\n";
    vector<int> v3 = {10, 20, 30};
    printVector_ConstRef(v3, "v3");
    cout << "sumVector_ConstRef(v3) = " << sumVector_ConstRef(v3) << "\n";

    // Works with temporaries too:
    printVector_ConstRef(vector<int>{9, 8, 7}, "temporary vector");
    cout << "\n";

    cout << "=== 4) Pass by pointer (*) ===\n";
    int c = 30;
    cout << "Before changeByPointer: c = " << c << "\n";
    changeByPointer(&c);
    cout << "After changeByPointer:  c = " << c << " (changed)\n";

    changeByPointer(nullptr); // safe: function checks for null

    string name = "Ada";
    printIfNotNull(&name);
    printIfNotNull(nullptr);
    cout << "\n";

    cout << "=== Quick rule of thumb ===\n";
    cout << "- Small types (int, double): pass by value\n";
    cout << "- Read-only big types (string, vector): pass by const &\n";
    cout << "- Need to modify: pass by & (or pointer if null is meaningful)\n";

    return 0;
}
