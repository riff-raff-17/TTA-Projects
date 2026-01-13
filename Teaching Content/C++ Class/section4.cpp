// ------------------------------------------------------
// 04_if_else.cpp
// Section 4: if, else if, and else
// ------------------------------------------------------

#include <iostream>
using namespace std;

int main() {

    int score;
    cout << "Enter your score (0 - 100): ";
    cin >> score;

    // ---------- Simple if ----------
    if (score >= 50) {
        cout << "You passed!" << endl;
    }

    // ---------- if / else ----------
    if (score >= 50) {
        cout << "Pass" << endl;
    } else {
        cout << "Fail" << endl;
    }

    // ---------- if / else if / else ----------
    if (score >= 80) {
        cout << "Grade: A" << endl;
    } else if (score >= 70) {
        cout << "Grade: B" << endl;
    } else if (score >= 60) {
        cout << "Grade: C" << endl;
    } else if (score >= 50) {
        cout << "Grade: D" << endl;
    } else {
        cout << "Grade: F" << endl;
    }

    cout << "------------------------" << endl;

    // ---------- Multiple conditions ----------
    int age;
    cout << "Enter your age: ";
    cin >> age;

    if (age >= 13 && age <= 19) {
        cout << "You are a teenager." << endl;
    } else {
        cout << "You are not a teenager." << endl;
    }

    cout << "------------------------" << endl;

    // ---------- NOT operator ----------
    bool isRaining;
    cout << "Is it raining? (1 = yes, 0 = no): ";
    cin >> isRaining;

    if (!isRaining) {
        cout << "You don't need an umbrella." << endl;
    } else {
        cout << "Bring an umbrella!" << endl;
    }

    // Short Hand If...Else (Ternary Operator)
    int time = 20;
    string result = (time < 18) ? "Good day." : "Good evening.";
    cout << result;

    return 0;
}
