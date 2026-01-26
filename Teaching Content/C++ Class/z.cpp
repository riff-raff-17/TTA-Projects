#include <algorithm> // sort, reverse, find, count, remove
#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main() {
    vector<int> v;
    
    // Increase the capacity of vector
    v.reserve(9);
    cout << v.capacity();

    int numbers[5];

    cout << "Array elements:" << endl;
    for (int i = 0; i < 5; i++) {
        cout << numbers[i] << " ";
    }

    string name = "Hello";
    cout << name[0];
  
    return 0;
}