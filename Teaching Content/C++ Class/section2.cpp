// Section 2: Variables, Data Types, and Input

#include <iostream>
using namespace std;

int main() {

    // ---------- Integer variables ----------
    int age = 0; // whole number
    int year = 2026;

    // ---------- Double (decimal) ----------
    double height = 0.0;

    // ---------- Character ----------
    // Notice the single quotes!
    char grade = 'A';

    // ---------- Boolean ----------
    bool isStudent = true;

    // ---------- String (text) ----------
    string name = "Unknown";

    // Print initial values
    cout << "Age: " << age << endl;
    cout << "Year: " << year << endl;
    cout << "Height: " << height << endl;
    cout << "Grade: " << grade << endl;
    cout << "Is student: " << isStudent << endl; // Notice the output!
    cout << "Name: " << name << endl;

    cout << "------------------------" << endl;

    // ---------- Input ----------
    cout << "Enter your name: ";
    cin >> name; // reads ONE word only!

    cout << "Enter your age: ";
    cin >> age;

    cout << "Enter your height (in meters): ";
    cin >> height;

    cout << "Enter your grade (A-F): ";
    cin >> grade;

    cout << "------------------------" << endl;

    // Print user input
    cout << "Hello, " << name << "!" << endl;
    cout << "You are " << age << " years old." << endl;
    cout << "Your height is " << height << " meters." << endl;
    cout << "Your grade is " << grade << endl;

    return 0;
}
