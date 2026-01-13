// Section 1: How C++ runs and Hello World

// This line includes the input/output library
#include <iostream>

// This allows us to use names like cout without writing std::cout
using namespace std;

// Every C++ program MUST have a main function
int main() {

    // Print text to the screen
    // MUST use double quotes
    // MUST end with ;
    cout << "Hello, C++!" << endl; // endl is endline. You could also do \n

    // Print another line
    cout << "My first C++ program" << endl;

    // You can also print numbers
    cout << 123 << endl;

    // You can have multiple lines in one cout!
    cout << "This\nis\nall\non\nnewlines" << endl;

    // ASCII Art
    cout << "****\n" << "*  *\n" << "*  *\n" << "****\n";

    cout << "*\n" << "**\n" << "***\n" << "****\n";

    // The program ends here
    // Returning 0 tells the computer "the program ran successfully"
    return 0;
}
