// STAGE 2 of 5 — Filtering candidates

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace wordle
{

    enum class Color
    {
        Gray,
        Yellow,
        Green
    };
    using Feedback = std::array<Color, 5>;

    Feedback computeFeedback(const std::string &guess, const std::string &answer)
    {
        Feedback result{};
        result.fill(Color::Gray);

        std::array<bool, 5> answerLetterUsed{};
        answerLetterUsed.fill(false);

        for (int i = 0; i < 5; ++i)
        {
            if (guess[i] == answer[i])
            {
                result[i] = Color::Green;
                answerLetterUsed[i] = true;
            }
        }

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

    // NEW in stage 2 --------------------------------------------------

    // True if `candidate` could still be the secret word, given that `guess`
    // produced `feedback`. The trick: if candidate WERE the answer, guessing
    // `guess` against it would have to reproduce the exact feedback we saw.
    bool isConsistent(const std::string &candidate, const std::string &guess,
                      const Feedback &feedback)
    {
        return computeFeedback(guess, candidate) == feedback;
    }

    // Returns the subset of `candidates` still consistent with a guess/feedback pair.
    std::vector<std::string> filterCandidates(const std::vector<std::string> &candidates,
                                              const std::string &guess,
                                              const Feedback &feedback)
    {
        std::vector<std::string> result;
        result.reserve(candidates.size());
        for (const auto &word : candidates)
        {
            if (isConsistent(word, guess, feedback))
            {
                result.push_back(word);
            }
        }
        return result;
    }

} // namespace wordle

// ------------------------------------------------------------------
// Test harness
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

    void printWords(const std::vector<std::string> &words)
    {
        for (const auto &w : words)
            std::cout << "  " << w << "\n";
    }

} // namespace

int main()
{
    std::cout << "=== Stage 2: candidate filtering ===\n\n";

    // A small, hand-picked pool small enough to reason about by eye.
    std::vector<std::string> candidates = {
        "world", "would", "words", "wooer", "worry",
        "sound", "round", "bound", "found", "mound",
        "crane", "plane", "grape", "shape", "spare"};

    std::cout << "Starting pool (" << candidates.size() << " words):\n";
    printWords(candidates);

    // Simulate having guessed "sound" against a secret word of "world":
    // s=gray, o=green, u=gray, n=gray, d=green. (We know this is the
    // right feedback for that pair because stage 1 already proved
    // computeFeedback correct — feel free to verify by hand too.)
    std::string guess = "sound";
    wordle::Feedback fb = wordle::computeFeedback(guess, "world");

    std::cout << "\nGuessed \"" << guess << "\" against a secret of \"world\" would give: "
              << feedbackToString(fb) << "\n";
    std::cout << "Filtering the pool down to only words consistent with that feedback:\n";

    auto filtered = wordle::filterCandidates(candidates, guess, fb);

    std::cout << "\nRemaining (" << filtered.size() << " word(s)):\n";
    printWords(filtered);

    std::cout << "\n\"world\" should be in that remaining list (since it's literally the\n";
    std::cout << "word we generated the feedback from) — any word sharing the same 'o'\n";
    std::cout << "in position 2, 'd' in position 5, and no 's'/'u'/'n' anywhere else\n";
    std::cout << "would also survive. If \"world\" is in the output, move on to stage 3.\n";

    return 0;
}
