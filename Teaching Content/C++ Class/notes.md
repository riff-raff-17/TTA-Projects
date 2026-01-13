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