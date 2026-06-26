# OOP Basics in C++ — 1-Hour Lesson: Talking Points

Companion file: `car.cpp` (live-code this alongside the talking points below).
The whole lesson uses one running example — a `Car` class — that evolves stage by stage.

---

## 0. Setup (before class starts)
- Have a blank `.cpp` file ready, or `car.cpp` open with later stages commented out.
- Compiler/IDE ready to run code instantly after each change — the payoff of live coding is seeing it work immediately.

---

## 1. Why OOP? (5 min)

**Talking points:**
- Without OOP, you'd have separate variables and functions that operate on them, with nothing tying them together. Easy to lose track of what data belongs with what logic.
- OOP bundles data (**attributes**) and behavior (**methods**) into one unit: an **object**.
- Analogy: a `Car` has properties (color, speed) and things it can do (accelerate, brake). In real life we don't think of "car" and "the car's speed" as separate things — OOP lets code mirror that.

**Say something like:**
> "Today we're building one thing — a `Car` — and we'll keep adding capabilities to it as we learn each concept. By the end, you'll have seen a whole class evolve in front of you."

---

## 2. Classes & Objects (15 min)

**Talking points:**
- A **class** is a blueprint. A "car" is not a real, physical car — it's a description of what a car has and does.
- An **object** is an actual instance built from that blueprint. You can make many objects from one class.
- Show the class definition (Stage 2 in `car.cpp`):
  ```cpp
  class Car {
  public:
      string color;
      int speed;

      void accelerate() {
          speed += 10;
          cout << "Vroom! Speed is now " << speed << endl;
      }
  };
  ```
- Create two objects, set different colors/speeds, call `accelerate()` on each — show that they don't interfere with each other. **Each object has its own copy of the data.**

**Live demo:**
```cpp
Car car1;
car1.color = "red";
car1.speed = 0;
car1.accelerate();

Car car2;
car2.color = "blue";
car2.speed = 0;
car2.accelerate();
car2.accelerate();
```

**Ask the audience:** "What do you think `car1.speed` will be after this? What about `car2.speed`?" — gets them predicting before you run it.

---

## 3. Access Specifiers & Encapsulation (10 min)

**Talking points:**
- Right now, anyone can do `car1.speed = -500;`. That's nonsense for a real car, but the compiler has no problem with it.
- **Encapsulation** = hiding the internal data and only allowing controlled access through methods.
- `private` members can only be touched from inside the class. `public` members/methods are the "front door."
- Introduce **getters** (read access) and **setters** (write access, often with validation).

**Live demo — evolve the class:**
```cpp
class CarEncapsulated {
private:
    int speed;
public:
    string color;

    void accelerate() {
        speed += 10;
        cout << "Vroom! Speed is now " << speed << endl;
    }

    int getSpeed() { return speed; }

    void setSpeed(int s) {
        if (s < 0) {
            cout << "Speed can't be negative! Ignoring." << endl;
            return;
        }
        speed = s;
    }
};
```
- Try `car3.speed = -100;` directly — show the compile error. Then show `car3.setSpeed(-100);` being safely rejected.

**Key line to say:**
> "Encapsulation isn't about secrecy for its own sake — it's about making invalid states impossible. The class protects itself."

---

## 4. Constructors & Destructors (15 min)

**Talking points:**
- Problem: right now, if you forget to set `color` and `speed`, you get garbage/default values. Easy to forget.
- A **constructor** runs automatically when an object is created — guarantees it starts in a valid state.
- Constructors can be **default** (no arguments) or **parameterized** (you pass in initial values).
- A **destructor** runs automatically when an object is destroyed (goes out of scope). Useful for cleanup — we won't need real cleanup here, but it's good to know it exists.

**Live demo — evolve again:**
```cpp
class CarWithConstructor {
private:
    int speed;
public:
    string color;

    CarWithConstructor() {
        color = "unknown";
        speed = 0;
        cout << "A car was created!" << endl;
    }

    CarWithConstructor(string c, int s) {
        color = c;
        speed = s;
        cout << "A " << color << " car was created!" << endl;
    }

    ~CarWithConstructor() {
        cout << "A " << color << " car was destroyed." << endl;
    }

    void accelerate() {
        speed += 10;
        cout << "Vroom! Speed is now " << speed << endl;
    }

    int getSpeed() { return speed; }
};
```
- Run it and point out the destructor messages printing automatically at the end of `main()` — nobody called them manually.

**Say something like:**
> "Notice we never call the destructor ourselves. C++ does it automatically. This is part of what makes C++ object lifetimes predictable — something we'll dig into more in a later lesson."

---

## 5. A Taste of Inheritance (10 min)

**Talking points:**
- So far every car-like thing has been a `Car`. What if we want a `SportsCar` that's mostly a car, but with an extra ability?
- **Inheritance** lets one class reuse another class's code. This models an **"is-a"** relationship: a sports car *is a* car.
- Keep this brief — full inheritance (virtual functions, multiple inheritance, etc.) is next lesson. Today, just plant the seed.

**Live demo:**
```cpp
class SportsCar : public CarWithConstructor {
public:
    SportsCar(string c, int s) : CarWithConstructor(c, s) {
        cout << "...and it's a SPORTS car!" << endl;
    }

    void turboBoost() {
        accelerate();
        accelerate();
        cout << "TURBO BOOST ENGAGED!" << endl;
    }
};
```
- Run it, show that `sc.accelerate()` works even though `SportsCar` never wrote that method itself — it inherited it.

**Key line to say:**
> "SportsCar got color, speed, the constructor logic, and accelerate() completely for free. All we added was turboBoost(). That's the power of inheritance — write once, reuse everywhere."

---

## 6. Wrap-up / Q&A (5 min)

**Recap out loud, in order:**
1. **Classes & objects** — blueprint vs. instance
2. **Encapsulation** — private data, public interface, getters/setters
3. **Constructors/destructors** — guaranteed valid setup and automatic cleanup
4. **Inheritance** — reusing code via "is-a" relationships

**Tease next lesson:**
> "Next time: polymorphism — how different objects can respond differently to the same instruction — plus more on inheritance, and a few other tools you'll want before writing real C++ programs."

**Leave them with a challenge:**
> "Before next time, try adding a `Truck` class that inherits from `CarWithConstructor`. Give it a `cargoWeight` variable and a `loadCargo(int weight)` method that adds to it."

---

## Timing cheat sheet

| Section                          | Minutes | Running total |
|-----------------------------------|---------|----------------|
| Why OOP?                          | 5       | 5              |
| Classes & Objects                 | 15      | 20             |
| Access Specifiers & Encapsulation | 10      | 30             |
| Constructors & Destructors        | 15      | 45             |
| Taste of Inheritance              | 10      | 55             |
| Wrap-up / Q&A                     | 5       | 60             |
