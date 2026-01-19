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
  
    return 0;
}