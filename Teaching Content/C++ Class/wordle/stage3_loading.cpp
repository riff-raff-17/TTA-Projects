// STAGE 3 of 5 — Loading the real word list

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
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

    bool isConsistent(const std::string &candidate, const std::string &guess,
                      const Feedback &feedback)
    {
        return computeFeedback(guess, candidate) == feedback;
    }

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

    // NEW in stage 3 --------------------------------------------------

    // Reads one word per line from 'path', keeping only clean 5-letter
    // entries (lowercased, trailing whitespace stripped).
    std::vector<std::string> loadWords(const std::string &path)
    {
        std::vector<std::string> words;
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line))
        {
            while (!line.empty() && !std::isalpha(static_cast<unsigned char>(line.back())))
            {
                line.pop_back();
            }
            if (line.size() == 5)
            {
                std::transform(line.begin(), line.end(), line.begin(), ::tolower);
                words.push_back(line);
            }
        }
        return words;
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

} // namespace

int main()
{
    std::cout << "=== Stage 3: loading the real word list ===\n\n";

    auto words = wordle::loadWords("words.txt");
    if (words.empty())
    {
        std::cerr << "Could not load words.txt. Make sure it's in this directory,\n"
                     "one 5-letter word per line.\n";
        return 1;
    }
    std::cout << "Loaded " << words.size() << " words from words.txt\n";

    // Sanity check: is the file actually 5-letter words, no stray junk?
    bool allFiveLetters = true;
    for (const auto &w : words)
    {
        if (w.size() != 5)
        {
            allFiveLetters = false;
            break;
        }
    }
    std::cout << "All entries exactly 5 letters: " << (allFiveLetters ? "yes" : "NO - check words.txt") << "\n";

    // Same round-trip idea as stage 2, but now against the full list:
    // pick a real word from the loaded list as a pretend secret answer,
    // guess something else, filter, and confirm the secret survives.
    std::string secret = words[words.size() / 2]; // an arbitrary real word from the list
    std::string guess = "sound";
    wordle::Feedback fb = wordle::computeFeedback(guess, secret);

    std::cout << "\nPretend secret: \"" << secret << "\"\n";
    std::cout << "Guessed \"" << guess << "\" -> feedback " << feedbackToString(fb) << "\n";

    auto filtered = wordle::filterCandidates(words, guess, fb);
    std::cout << "Candidates remaining: " << filtered.size() << " (out of " << words.size() << ")\n";

    bool secretSurvived = std::find(filtered.begin(), filtered.end(), secret) != filtered.end();
    std::cout << "Secret word still in the filtered list: " << (secretSurvived ? "YES" : "NO - bug!") << "\n";

    std::cout << "\nIf the secret survived and the candidate count dropped well below "
              << words.size() << ",\nfiltering is working correctly at full scale. Move on to stage 4.\n";

    return 0;
}
