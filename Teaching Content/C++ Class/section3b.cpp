// ------------------------------------------------------
// 03_operators.cpp
// Section 3: Operators & Expressions
// ------------------------------------------------------

#include <iostream>
using namespace std;

int main() {

    int a = 10;
    int b = 3;

    // ---------- Arithmetic ----------
    cout << "a + b = " << a + b << endl;
    cout << "a - b = " << a - b << endl;
    cout << "a * b = " << a * b << endl;
    cout << "a / b = " << a / b << endl; // integer division
    cout << "a % b = " << a % b << endl;

    cout << "------------------------" << endl;

    // ---------- Decimal division ----------
    double x = 10;
    double y = 3;

    cout << "x / y = " << x / y << endl;

    cout << "------------------------" << endl;

    // ---------- Comparisons ----------
    cout << "a == b: " << (a == b) << endl;
    cout << "a != b: " << (a != b) << endl;
    cout << "a > b:  " << (a > b) << endl;
    cout << "a < b:  " << (a < b) << endl;

    cout << "------------------------" << endl;

    // ---------- Logical operators ----------
    bool cond1 = (a > 5);
    bool cond2 = (b < 5);

    cout << "cond1 AND cond2: " << (cond1 && cond2) << endl;
    cout << "cond1 OR cond2:  " << (cond1 || cond2) << endl;
    cout << "NOT cond1:       " << (!cond1) << endl;

    cout << "------------------------" << endl;

    // ---------- Parentheses matter ----------
    cout << "Without parentheses: " << a + b * 2 << endl;
    cout << "With parentheses:    " << (a + b) * 2 << endl;

    return 0;
}
