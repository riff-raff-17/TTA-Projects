// 06_functions.cpp
// Section 6: Functions

#include <iostream>
using namespace std;

// Function declarations

// Returns the sum of two integers
int add(int a, int b) {
    return a + b;
}

// Prints a greeting
void greet() {
    cout << "Hello from a function!" << endl;
}

// Checks if a number is even
bool isEven(int n) {
    return n % 2 == 0;
}

// Returns the larger of two numbers
int maxOfTwo(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int main() {

    // Calling functions 
    greet();

    int x = 5;
    int y = 8;

    int sum = add(x, y);
    cout << "Sum: " << sum << endl;

    cout << "Is " << x << " even? " << isEven(x) << endl;
    cout << "Max of x and y: " << maxOfTwo(x, y) << endl;

    cout << "------------------------" << endl;

    // Pass-by-value demonstration 
    int number = 10;
    cout << "Before function: " << number << endl;

    // Try to change number
    add(number, 5);

    cout << "After function: " << number << endl;

    return 0;
}
