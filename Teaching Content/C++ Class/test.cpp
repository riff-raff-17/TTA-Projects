#include <iostream>
using namespace std;

int main() {
    int a;
    string output;

    cin >> a;

    if (a % 3 == 0){
        output += "Divisible by 3";
        if (a % 5 == 0){
            output += " and 5";
        }
    }
    else if (a % 5 == 0){
        output += "Divisible by 5";
    }
    else {
        output += "Not divisible";
    }

    cout << output << endl;

    return 0;
}
