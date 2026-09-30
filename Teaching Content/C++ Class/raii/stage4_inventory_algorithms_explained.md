# Stage 4: Inventory Algorithms — In-Depth Walkthrough

**File:** `stage4_inventory_algorithms.cpp`
**Lesson section:** 6 — Inventory Management with STL Algorithms
**Compile:** `g++ -std=c++17 -o stage4_inventory_algorithms stage4_inventory_algorithms.cpp`

---

## What this stage is for

Everything from Stage 3 (items, enemies, combat) is carried over unchanged. This stage
adds exactly three new `Player` methods, all built on `<algorithm>` and `<numeric>`,
that turn `inventory` from "a list you can only push/find/remove one at a time" into
something you can meaningfully summarize and reorder. This is the shortest diff of any
stage — the point isn't new architecture, it's demonstrating that once you have a
`vector<unique_ptr<T>>`, the whole STL algorithm toolkit is available to you, with one
recurring constraint to internalize.

---

## `sortInventoryByValue()`

```cpp
void sortInventoryByValue() {
    sort(inventory.begin(), inventory.end(),
        [](const unique_ptr<Item>& a, const unique_ptr<Item>& b) {
            return a->getValue() > b->getValue();
        });
    cout << "Inventory sorted by value (highest first)." << endl;
}
```

- **`sort` takes a begin/end iterator pair and a comparator.** The comparator answers
  "should `a` come before `b`?" — here, `a->getValue() > b->getValue()` means higher
  value sorts first (descending order). Flip the `>` to `<` and you'd get ascending
  order instead.
- **The comparator parameters are `const unique_ptr<Item>&`, never `unique_ptr<Item>`
  by value.** This isn't a style preference — it's a hard requirement. `unique_ptr`
  cannot be copied (only moved), and a by-value lambda parameter would require copying
  each element to call the lambda, which simply won't compile. Using a `const&`
  reference means the comparator only ever *looks* at each element to compare it,
  never takes ownership of it.
- **What `sort` is actually doing under the hood: moving, not copying.** When `sort`
  rearranges `inventory`'s elements into the new order, it does so by moving the
  `unique_ptr` objects themselves (cheap — just reassigning the internal raw pointer),
  never by copying the `Item` objects they point to. This is the same move-semantics
  idea from earlier lessons, now happening automatically inside a standard algorithm
  rather than something you write explicitly with `move()`.

---

## `totalInventoryValue()`

```cpp
int totalInventoryValue() const {
    return accumulate(inventory.begin(), inventory.end(), 0,
        [](int sum, const unique_ptr<Item>& i) { return sum + i->getValue(); });
}
```

- **`accumulate` folds a range down to a single value.** Its three required arguments
  are: the range (`begin`, `end`), a starting value (`0`), and — when you pass a fourth
  argument — a custom combining function instead of the default `+`. Without that
  fourth argument, `accumulate` would try to add `unique_ptr<Item>` objects directly
  with `+`, which doesn't compile (there's no `operator+` for `unique_ptr`). The lambda
  here is what makes it possible: it takes the running `sum` (a plain `int`, passed by
  value — cheap to copy) and one inventory element (by `const&`, for the same reason as
  `sort`'s comparator), and returns the updated sum.
- **The lambda's first parameter type (`int sum`) matches the starting value's type
  (`0`, an `int`).** `accumulate` is a template function; the type it folds into is
  inferred from that starting value, so if you passed `0.0` instead of `0`, the lambda
  would need to take a `double` instead. This detail is easy to overlook and worth
  pointing out live if anyone tries to change the starting value.

---

## `countPotions()`

```cpp
int countPotions() const {
    return static_cast<int>(count_if(inventory.begin(), inventory.end(),
        [](const unique_ptr<Item>& i) { return dynamic_cast<Potion*>(i.get()) != nullptr; }));
}
```

- **`count_if` counts how many elements satisfy a predicate**, without needing to
  build a separate filtered list first. It returns the count as the container's
  `difference_type` (effectively a signed integer type), hence the
  `static_cast<int>(...)` wrapping the whole call — a small, explicit conversion to
  match this function's declared `int` return type.
- **`dynamic_cast<Potion*>(i.get())` is how the predicate checks the item's actual
  subtype.** `i.get()` extracts the raw `Item*` from the `unique_ptr` (again, no
  ownership change — just a temporary view). `dynamic_cast` then asks "at runtime, is
  the object this pointer refers to *actually* a `Potion`, or some other `Item`
  subtype?" If it is, the cast succeeds and returns a valid `Potion*`; if not, it
  returns `nullptr`. Comparing the result to `nullptr` turns that into a yes/no answer
  for the predicate.
- **Why `dynamic_cast` here and not, say, a `type` enum field on `Item`?** `dynamic_cast`
  uses the polymorphic type information C++ already tracks for any class with virtual
  functions (which `Item` has, via `use()` and `describe()`) — there's no need to
  invent and maintain a separate "what kind of item is this" flag by hand. It does have
  a small runtime cost compared to a plain flag check, which is worth mentioning as a
  tradeoff, but for a case like this — an occasional inventory query, not a hot loop —
  it's the right tool.

---

## The `status` and `sort` commands in `main()`

```cpp
else if (cmd == "sort") {
    player.sortInventoryByValue();
} else if (cmd == "status") {
    cout << "HP: " << player.getHealth() << "/" << player.getMaxHealth() << endl;
    cout << "Inventory value: " << player.totalInventoryValue() << endl;
    cout << "Potions carried: " << player.countPotions() << endl;
}
```

Nothing new mechanically here — these two commands just expose the three new methods
directly. Worth noticing: `status` calls all three of `getHealth()`,
`totalInventoryValue()`, and `countPotions()` back-to-back, each independently
iterating over `inventory` (for the two algorithm-based ones). For an inventory this
small, running three separate passes over the vector is completely fine; it's a detail
worth flagging for later (a larger, more performance-sensitive project might combine
several of these into a single pass), but not something to over-engineer here.

## Key concepts this stage teaches

1. **`sort`, `accumulate`, and `count_if`** — three of the STL's most commonly used
   algorithms, each solving a different shape of problem: reordering, folding to one
   value, and conditional counting.
2. **Every algorithm operating on `unique_ptr` elements takes them by `const&` in its
   lambda** — a rule that follows directly from `unique_ptr` not being copyable, and
   applies no matter which algorithm you're using.
3. **`accumulate`'s starting-value argument determines the fold type** — a detail that
   only becomes visible once you try to change it.
4. **`dynamic_cast`** as the tool for "what subtype is this, specifically?" — distinct
   from virtual dispatch (`item->use()`), which is for "run the right behavior without
   needing to know the subtype at all."

## Try it

- `take` all four starting items, `sort`, then `inventory` — confirm the Elixir (worth
  60) appears above the Iron Sword (worth 50).
- `status` before and after dropping (using) a potion — confirm the potion count and
  total value both update correctly.
- Add a `Scroll` item type (a fourth `Item` subclass) and confirm `countPotions()`
  correctly does *not* count it, since `dynamic_cast<Potion*>` on a `Scroll` returns
  `nullptr`.
