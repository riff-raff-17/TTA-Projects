// STAGE 1 of 5 — The feedback engine

#include <array>
#include <iostream>
#include <string>

namespace wordle
{

    // A single tile's result. Gray = letter not in word, Yellow = in word but
    // wrong spot, Green = correct letter in the correct spot.
    enum class Color
    {
        Gray,
        Yellow,
        Green
    };
    using Feedback = std::array<Color, 5>;

    // Computes the feedback pattern Wordle would show for `guess` if the
    // secret word were `answer`. Handles duplicate letters the way the real
    // game does: greens are claimed first, then yellows use what's left.
    Feedback computeFeedback(const std::string &guess, const std::string &answer)
    {
        Feedback result{};
        result.fill(Color::Gray);

        std::array<bool, 5> answerLetterUsed{};
        answerLetterUsed.fill(false);

        // Pass 1: exact position matches (greens) claim their letter first.
        for (int i = 0; i < 5; ++i)
        {
            if (guess[i] == answer[i])
            {
                result[i] = Color::Green;
                answerLetterUsed[i] = true;
            }
        }

        // Pass 2: remaining letters get yellow if unclaimed in the answer.
        for (int i = 0; i < 5; ++i)
        {
            if (result[i] == Color::Green)
                continue;
            for (int j = 0; j < 5; ++j)
            {
                if (!answerLetterUsed[j] && guess[i] == answer[j])
                {
                    result[i] = Color::Yellow;
                    answerLetterUsed[j] = true;
                    break;
                }
            }
        }

        return result;
    }

} // namespace wordle

// ------------------------------------------------------------------
// Test harness — not part of the "real" program, just here to prove
// stage 1 works before you build stage 2 on top of it.
// ------------------------------------------------------------------

namespace
{

    std::string feedbackToString(const wordle::Feedback &fb)
    {
        std::string s;
        for (auto c : fb)
        {
            s += (c == wordle::Color::Green) ? 'G' : (c == wordle::Color::Yellow) ? 'Y'
                                                                                  : 'B';
        }
        return s;
    }

    void check(const std::string &guess, const std::string &answer, const std::string &expected)
    {
        std::string actual = feedbackToString(wordle::computeFeedback(guess, answer));
        bool pass = (actual == expected);
        std::cout << (pass ? "PASS   " : "FAIL   ")
                  << guess << " vs " << answer
                  << "  expected " << expected << "  got " << actual << "\n";
    }

} // namespace

int main()
{
    std::cout << "=== Stage 1: feedback engine ===\n\n";

    // Straightforward case: no repeated letters.
    check("crane", "grape", "BGGBG");

    // The tricky case: guess has three S's, answer only has two.
    // Greens claim first (position 3), leaving one unclaimed 's' in the
    // answer for the two remaining guess S's to compete over.
    check("sassy", "abyss", "YYBGY");

    // All correct.
    check("world", "world", "GGGGG");

    // No overlap at all.
    check("chimp", "world", "BBBBB");

    // Guess has two O's, answer has two O's in different spots: the
    // first O lands on a green match (position 1); the second O then
    // finds the answer's other, still-unclaimed O elsewhere -> yellow.
    check("robot", "moose", "BGBYB");

    std::cout << "\nIf every line above says PASS, stage 1 is solid — move on to stage 2.\n";
    return 0;
}
