// 05_loops.cpp
// Section 5: Loops (while, for, do-while)

#include <iostream>
using namespace std;

int main() {

    // ---------- while loop ----------
    int count = 1;

    while (count <= 5) {
        cout << "Count: " << count << endl;
        count++;
    }

    cout << "------------------------" << endl;

    // ---------- for loop ----------
    for (int i = 1; i <= 5; i++) {
        cout << "i: " << i << endl;
    }

    cout << "------------------------" << endl;

    // ---------- Sum from 1 to n ----------
    int n;
    cout << "Enter a number n: ";
    cin >> n;

    int sum = 0;

    for (int i = 1; i <= n; i++) {
        sum += i;
    }

    cout << "Sum from 1 to " << n << " is " << sum << endl;

    cout << "------------------------" << endl;

    // ---------- Factorial using while ----------
    int factN;
    cout << "Enter a number for factorial: ";
    cin >> factN;

    int factorial = 1;
    int i = 1;

    while (i <= factN) {
        factorial *= i;
        i++;
    }

    cout << "Factorial: " << factorial << endl;

    cout << "------------------------" << endl;

    // ---------- do-while ----------
    int choice;

    do {
        cout << "Enter a number between 1 and 5: ";
        cin >> choice;
    } while (choice < 1 || choice > 5);

    cout << "You entered a valid number!" << endl;

    return 0;
}
