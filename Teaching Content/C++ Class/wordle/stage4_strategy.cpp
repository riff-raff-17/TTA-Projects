// STAGE 4 of 5 — Picking a good next guess

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

    // NEW in stage 4 --------------------------------------------------

    // Picks the best next guess using letter-position frequency scoring.
    // This is the one function you'd replace to try a smarter algorithm
    // (like entropy-based guessing) later — everything else stays the same.
    std::string bestGuess(const std::vector<std::string> &candidates)
    {
        if (candidates.empty())
            return "";

        std::array<std::array<int, 26>, 5> freq{};
        for (const auto &word : candidates)
        {
            for (int i = 0; i < 5; ++i)
            {
                freq[i][word[i] - 'a']++;
            }
        }

        auto scoreWord = [&](const std::string &word)
        {
            double score = 0.0;
            std::array<bool, 26> seenLetter{};
            for (int i = 0; i < 5; ++i)
            {
                int letter = word[i] - 'a';
                score += freq[i][letter];
                if (seenLetter[letter])
                    score *= 0.5; // repeats add less info
                seenLetter[letter] = true;
            }
            return score;
        };

        std::string best = candidates[0];
        double bestScore = -1.0;
        for (const auto &word : candidates)
        {
            double s = scoreWord(word);
            if (s > bestScore)
            {
                bestScore = s;
                best = word;
            }
        }
        return best;
    }

} // namespace wordle

int main()
{
    std::cout << "=== Stage 4: guessing strategy ===\n\n";

    auto words = wordle::loadWords("words.txt");
    if (words.empty())
    {
        std::cerr << "Could not load words.txt\n";
        return 1;
    }
    std::cout << "Loaded " << words.size() << " words.\n";

    std::string firstGuess = wordle::bestGuess(words);
    std::cout << "Suggested first guess: " << firstGuess << "\n";

    // Simulate one round against a real secret word and confirm the
    // suggestion gets sharper as the pool shrinks.
    std::string secret = "world";
    wordle::Feedback fb = wordle::computeFeedback(firstGuess, secret);
    auto remaining = wordle::filterCandidates(words, firstGuess, fb);

    std::cout << "\nPretending the secret is \"" << secret << "\":\n";
    std::cout << "  " << remaining.size() << " candidate(s) remain after that guess\n";

    if (!remaining.empty())
    {
        std::string secondGuess = wordle::bestGuess(remaining);
        std::cout << "  Suggested next guess: " << secondGuess << "\n";
    }

    std::cout << "\nSanity check: bestGuess should always return a real word from the\n";
    std::cout << "pool it was given, and the candidate count should shrink noticeably\n";
    std::cout << "after each guess. If that held here, move on to stage 5 — the final,\n";
    std::cout << "fully interactive version.\n";

    return 0;
}
