// ------------------------------------------------------
// 08_structs.cpp
// Section 8: Structs
// ------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>
using namespace std;

// ---------- Struct definition ----------
struct Student {
    string name;
    int age;
    double score;
};

// ---------- Function using a struct ----------
void printStudent(Student s) {
    cout << "Name: " << s.name << endl;
    cout << "Age: " << s.age << endl;
    cout << "Score: " << s.score << endl;
}

int main() {

    // ---------- Single struct ----------
    Student s1;
    s1.name = "Alice";
    s1.age = 15;
    s1.score = 92.5;

    printStudent(s1);

    cout << "------------------------" << endl;

    // ---------- Vector of structs ----------
    vector<Student> students;

    int n;
    cout << "How many students? ";
    cin >> n;

    for (int i = 0; i < n; i++) {
        Student s;

        cout << "Enter name: ";
        cin >> s.name;

        cout << "Enter age: ";
        cin >> s.age;

        cout << "Enter score: ";
        cin >> s.score;

        students.push_back(s);
    }

    cout << "------------------------" << endl;

    // ---------- Processing vector of structs ----------
    double totalScore = 0;

    for (Student s : students) {
        printStudent(s);
        cout << endl;
        totalScore += s.score;
    }

    cout << "Average score: " << totalScore / students.size() << endl;

    return 0;
}
