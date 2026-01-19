#include <iostream>
using namespace std;

void changeValue(int x) {
    x = 10;   // modifies only the copy
}

int main() {
    int a = 5;
    changeValue(a);
    cout << a;   // Output: 5
    return 0;
}