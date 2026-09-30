# Section 1

- C++ is a compiled language
- We write source code (.cpp)
- The compiler translates it into a program the computer can run
- Every C++ program starts running at main()
- If there is no main(), the program cannot run
- Statements end with semicolons
- Missing ; is the most common beginner error
- Curly braces {} define blocks
- Everything inside main runs in order, top to bottom

# Section 2

## What is a variable?

A variable:

- Stores a value in memory
- Has a type
- Has a name

| Type     | Example values | Used for         |
| -------- | -------------- | ---------------- |
| `int`    | `-3, 0, 42`    | whole numbers    |
| `double` | `3.14, 2.5`    | decimals         |
| `char`   | `'A', 'z'`     | single character |
| `bool`   | `true, false`  | yes / no         |
| `string` | `"Hello"`      | text             |

Input with cin:

- cin reads input from the keyboard
- Input is separated by spaces
- The value must match the variable’s type

### Why does bool print as 1 or 0?

true = 1

false = 0
(We’ll fix formatting later — this is normal.)

### Why cin >> name only reads one word?

cin stops at spaces

Full sentences need `getline()`! What's that? We'll touch on that later.

## Tasks
**Task A**

Modify the program to ask for:
- Favourite number (int)
- Likes programming (bool)

Print them nicely.

**Task B**

Get the year you were born as input, and output the age you will turn this year.

**Task C**

Read two **integers** and print:
- Sum
- Difference
- Product
- Quotient

What do you notice about the output?

What happens if I input 2.5 into an `int`?

# Section 3

**Arithmetic operators**

| Operator | Meaning        | Example |
| -------- | -------------- | ------- |
| `+`      | addition       | `a + b` |
| `-`      | subtraction    | `a - b` |
| `*`      | multiplication | `a * b` |
| `/`      | division       | `a / b` |
| `%`      | remainder      | `a % b` |

`%` only works with integers.

**Comparison operators**

| Operator | Meaning          |
| -------- | ---------------- |
| `==`     | equal            |
| `!=`     | not equal        |
| `<`      | less than        |
| `>`      | greater than     |
| `<=`     | less or equal    |
| `>=`     | greater or equal |

`=` is assignment, not comparison (Same as Python).

**Logical operators**
| Operator | Meaning |
| -------- | ------- |
| `&&`     | AND     |
| `\|\|`   | OR      |
| `!`      | NOT     |

## Tasks

**Task A**

Read in an integer `n`. Print 1 if `n` is between 10 and 20 inclusive. 0 otherwise.

**Task B**

Read in an integer `n`. Print 1 if `n` is even, `0` if odd.

(Hint: use %)

**Task C**

Read in three integers a, b, c.
Print 1 if all three are different, otherwise 0.

**Final question**

What is the value of 10 / 3 * 3?

# Section 4

## What is an `if` statement?

An `if` statement:

- Checks a condition
- Runs code only if the condition is true

Example: 
```cpp
if (age >= 18) {
    cout << "Adult";
}
```

Conditions must be true or false. Conditions usually come from:

- Comparisons (> < ==)
- Logical operators (&& || !)

## else and else if

if → first check

else if → checked only if previous was false

else → runs if nothing else matched

Example:
```cpp
    if (score >= 90) {
        cout << "Grade: A" << endl;
    }
    else if (score >= 75) {
        cout << "Grade: B" << endl;
    }
    else if (score >= 60) {
        cout << "Grade: C" << endl;
    }
    else {
        cout << "Grade: F" << endl;
    }
```

**Be careful!** Depending on you set up your conditions, some may *never* run.

What is wrong in the below example?

Example:
```cpp
    if (age >= 18) {
        cout << "You are an adult." << endl;
    }
    else if (age >= 65) {
        cout << "You are a senior." << endl;
    }
    else {
        cout << "You are a minor." << endl;
    }
```

Why this is wrong

The condition age >= 18 is **too general**. When age is 65 or higher, the first if is already true. The else if (age >= 65) is never reached.

So: Age 70 → prints “You are an adult”. “You are a senior” never happens.

> Always put the most specific condition first.
> Broad conditions should come later.

## Tasks


**Task A**

Read an integer n.

- Print "Positive" if n > 0
- Print "Negative" if n < 0
- Print "Zero" otherwise.

**Task B**

Read three integers.
Print the largest number.

(Hint: use if / else if OR nested if)

**Task C**
Write a C++ program that reads an integer x and prints:
- "Divisible by 3" if x is divisible by 3
- "Divisible by 5" if x is divisible by 5
- "Divisible by 3 and 5" if x is divisible by 5 AND 3
- "Not divisible" if x is not divisible by 3 or 5

**Task D**
What will the output of this program be?

```cpp
int a = 5;
int b = 10;

if (a = b) {
    cout << "Equal" << endl;
} else {
    cout << "Not equal" << endl;
}
```

**Final question**

Why must score >= 80 come before score >= 50?


# Section 5

## Loops (while, for, do-while)

Loops allow us to:

- Repeat code without copying it
- Count
- Accumulate values (sum, product)
- Process ranges of numbers

Without loops:

- Programs would be long and repetitive
- Many problems would be impossible to solve cleanly

## The while loop

A while loop:

- Checks the condition before each iteration
- May run zero times

Structure:
```cpp
while (condition) {
    // repeated code
}
```

For example:
```cpp
int i = 0;
while (i < 5) {
  cout << i << "\n";
  i++;
}
```

## The for loop


A for loop is best when:
- You know how many times to repeat

Structure:

```cpp
for (initialization; condition; update) {
    // repeated code
}
```

For example:
```cpp
for (int i = 0; i < 5; i++) {
  cout << i << "\n";
}
```

## The do-while loop

A do-while loop:
- Runs at least once
- Checks the condition after the loop body

Structure:

```cpp
do {
  // code block to be executed
}
while (condition);
```

For example:
```cpp
int i = 0;
do {
  cout << i << "\n";
  i++;
}
while (i < 5);
```

## Important notes

** The loop condition must eventually become false**

Otherwise, the loop runs forever.

Example of a bug:

```cpp
int x = 1;
while (x <= 5) {
    cout << x;
    // missing x++
}
```

**When to use each loop:**

- while → unknown number of repetitions
- for → known number of repetitions
- do-while → must run at least once

**Off-by-one errors**

These are the most common loop bugs.

```cpp
for (int i = 0; i < 5; i++)
```

## Tasks

**Task A**
Print all numbers from:
- 1 to 20 inclusive
- On one line, separated by spaces

**Task B**

Read an integer n.
Print all even numbers from 1 to n.

**Task C**

Read an integer n.
Print the sum of even numbers from 1 to n.

**Task D**

Read an integer n.
Print a square of stars of size n.

Example for n = 4:
```cpp
****
****
****
****
```

**Task E**

Read an integer n.
Print a triangle:

For n = 4:
```cpp
*
**
***
****
```

# Section 6

## Why functions exist

Functions allow us to:
- Reuse code
- Break problems into smaller pieces
- Make programs easier to read and debug

## Function structure

The basic structure of a section is:
```cpp
return_type function_name(parameters) {
    // code
    return value;
}
```

For example:
```cpp
int add(int a, int b) {
    return a + b;
}
```

## Void functions

Do **not** return a value

Used for printing or actions.

For example:

```cpp
void Greeter(string name){
    cout << "Hello, " << name << "!" << endl;
}
```

## Pass-by-value (important!)

A **copy** of the variable is passed

Changes inside the function do NOT affect the original

We’ll fix this later with references.

For example:
```cpp
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
```

## Function notes

### Function execution flow

When a function is called:
- Program jumps to the function
- Executes its code
- Returns to where it was called

### Why number does NOT change:


```cpp
add(number, 5);
```
number is copied into parameter a

The original variable is untouched

This is pass-by-value.

### `return` ends the function

Code after `return` is never executed.

## Tasks

**Task A**

Write a function:

```cpp
int square(int n)
```
Returns `n * n`.

**Task B**

Write a function:

```cpp
bool isPositive(int n)
```
Returns `true` if `n > 0`.

**Task C**

Write a function:

```cpp
void printStars(int n)
```

that prints `n` stars (*) on one line.

**Task D**

Write a function:
```cpp
int sumToN(int n)
````

Returns `1 + 2 + ... + n`.

**Task E**

Write a function:

```cpp
bool isPrime(int n)
```

# Section 7

## Arrays, Strings, and Introduction to `vector`

An array:

- Stores multiple values of the same type
- Has a fixed size
- Uses indexing starting at 0

For example:

```cpp
char a[5] = {'a', 'b', 'c', 'd', 'e'};
```

Indexes:
```cpp
a[0] a[1] a[2] a[3] a[4]

```


A string:
- Stores text
- Is basically a sequence of characters
- Is safer and easier than char[]

`getline`
- `cin >>` stops at spaces
- `getline` reads the whole line
- Must clear leftover newline first



A vector:
- Is a **resizable** array
- Knows its own size
- Is **safer** than arrays

Use vectors unless you have a reason not to.

| Array      | Vector       |
| ---------- | ------------ |
| Fixed size | Dynamic size |
| No size()  | Has size()   |
| Unsafe     | Safer        |
| Old C++    | Modern C++   |

## Tasks

**Task A**

Create an array of size 10. Read 10 integers and print the largest value.

**Task B**

Read a sentence. Count how many vowels it contains.

**Task C**

Use a vector to read `n` numbers. Print how many are even.

**Task D**

Use a vector to read `n` numbers. Print the numbers in reverse order.

**Task E**

Read words into a vector until the user enters "stop". Then print all words.


# Section 8

So far, we've used:

```cpp
string name;
int age;
double score;
```

But these variables belong together.

A struct lets us bundle related data into one unit.

## Struct definition

```cpp
struct Student {
    string name;
    int age;
    double score;
};
```

This creates a new type: `Student`.

## Accessing members

```cpp
Student s;
s.name = "Alex";
s.age = 14;
s.score = 87.5;
```