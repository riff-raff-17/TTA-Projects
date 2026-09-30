#include <iostream>
using namespace std;

// A pointer just holds a memory address of another variable.
void basics()
{
    cout << "\n--- 1. Basics: address-of (&) and dereference (*) ---\n";

    int x = 42;
    int *p = &x; // p now stores the ADDRESS of x

    cout << "x        = " << x << endl;
    cout << "&x       = " << &x << "  (address of x)" << endl;
    cout << "p        = " << p << "  (p holds that same address)" << endl;
    cout << "*p       = " << *p << "  (dereference p: 'the value p points to')" << endl;

    *p = 100; // changing *p changes x, since p points AT x
    cout << "after *p = 100, x is now: " << x << endl;
}

// A pointer that doesn't point anywhere valid should be nullptr.
void nullPointers()
{
    cout << "\n--- 2. nullptr ---\n";

    int *p = nullptr; // p points to nothing
    if (p == nullptr)
    {
        cout << "p is null - safe to check before using it" << endl;
    }
    // Dereferencing a null pointer (*p) would crash the program. Always check first.
}

// Pointers let a function modify the caller's variable directly ("pass by pointer").
void addTen(int *value)
{
    *value += 10; // follow the pointer, change what it points to
}

void passByPointer()
{
    cout << "\n--- 3. Passing a pointer to a function ---\n";

    int num = 5;
    cout << "before: " << num << endl;
    addTen(&num); // send the address of num
    cout << "after:  " << num << endl;
}

// Arrays and pointers are close cousins: an array name "decays" into
// a pointer to its first element.
void pointersAndArrays()
{
    cout << "\n--- 4. Pointers and arrays ---\n";

    int arr[5] = {10, 20, 30, 40, 50};
    int *p = arr; // same as: int* p = &arr[0];

    for (int i = 0; i < 5; i++)
    {
        // p[i] and *(p + i) and arr[i] all mean the same thing
        cout << "p[" << i << "] = " << p[i] << endl;
    }
}

// "new" allocates memory on the heap; you own it and must "delete" it yourself.
void dynamicMemory()
{
    cout << "\n--- 5. Dynamic memory (new / delete) ---\n";

    int *p = new int(7); // allocate a single int on the heap, initialized to 7
    cout << "*p = " << *p << endl;

    delete p;    // free the memory - required, or it's a memory leak
    p = nullptr; // good habit: avoid leaving a "dangling" pointer

    // Same idea for an array:
    int *arr = new int[3]{1, 2, 3};
    cout << "arr[1] = " << arr[1] << endl;
    delete[] arr; // note the [] - must match how it was allocated
    arr = nullptr;
}

int main()
{
    basics();
    nullPointers();
    passByPointer();
    pointersAndArrays();
    dynamicMemory();
}
