// 07_arrays_strings_vectors.cpp
// Section 7: Arrays, Strings, and Vectors

#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {

    // ---------- Arrays ----------
    int numbers[5] = {10, 20, 30, 40, 50};

    cout << "Array elements:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << numbers[i] << " ";
    }
    cout << endl;

    cout << "------------------------" << endl;

    // ---------- Strings ----------
    string name;

    cout << "Enter your full name: ";

    getline(cin, name);

    cout << "Hello, " << name << "!" << endl;
    cout << "Length of name: " << name.length() << endl;

    cout << "------------------------" << endl;

    // ---------- Vector ----------
    vector<int> values;

    cout << "Enter 5 integers:" << endl;
    for (int i = 0; i < 5; i++) {
        int x;
        cin >> x;
        values.push_back(x);
    }

    cout << "Vector contents:" << endl;
    for (int i = 0; i < values.size(); i++) {
        cout << values[i] << " ";
    }
    cout << endl;

    cout << "------------------------" << endl;

    // ---------- Vector processing ----------
    int sum = 0;
    for (int i = 0; i < values.size(); i++) {
        sum += values[i];
    }

    cout << "Sum of values: " << sum << endl;

    return 0;
}
