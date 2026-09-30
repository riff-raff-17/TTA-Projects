// wordle_solver.cpp

#include <array>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace wordle {

// ============================================================
// 1. ENGINE
// ============================================================

// A single tile's result. Gray = letter not in word, Yellow = in word but
// wrong spot, Green = correct letter in the correct spot.
enum class Color { Gray, Yellow, Green };
using Feedback = std::array<Color, 5>;

// Computes the feedback pattern Wordle would show for `guess` if the
// secret word were `answer`. Handles duplicate letters the way the real
// game does: greens are claimed first, then yellows use what's left.
// (Good test case: computeFeedback("sassy", "abyss") -> Y Y B G Y)
inline Feedback computeFeedback(const std::string& guess, const std::string& answer) {
    Feedback result{};
    result.fill(Color::Gray);

    std::array<bool, 5> answerLetterUsed{};
    answerLetterUsed.fill(false);

    // Pass 1: exact position matches (greens) claim their letter first.
    for (int i = 0; i < 5; ++i) {
        if (guess[i] == answer[i]) {
            result[i] = Color::Green;
            answerLetterUsed[i] = true;
        }
    }

    // Pass 2: remaining letters get yellow if unclaimed in the answer.
    for (int i = 0; i < 5; ++i) {
        if (result[i] == Color::Green) continue;
        for (int j = 0; j < 5; ++j) {
            if (!answerLetterUsed[j] && guess[i] == answer[j]) {
                result[i] = Color::Yellow;
                answerLetterUsed[j] = true;
                break;
            }
        }
    }

    return result;
}

// True if `candidate` could still be the secret word, given that `guess`
// produced `feedback`.
inline bool isConsistent(const std::string& candidate, const std::string& guess,
                          const Feedback& feedback) {
    return computeFeedback(guess, candidate) == feedback;
}

// Returns the subset of `candidates` still consistent with a guess/feedback pair.
inline std::vector<std::string> filterCandidates(const std::vector<std::string>& candidates,
                                                   const std::string& guess,
                                                   const Feedback& feedback) {
    std::vector<std::string> result;
    result.reserve(candidates.size());
    for (const auto& word : candidates) {
        if (isConsistent(word, guess, feedback)) {
            result.push_back(word);
        }
    }
    return result;
}

// ============================================================
// 2. STRATEGY
// ============================================================

// Picks the best next guess using letter-position frequency scoring:
// words made of common letters (in common positions) among the
// remaining candidates score higher. Everything else in this file is
// agnostic to how bestGuess works internally, so this is the one
// function to swap out for a smarter algorithm later.
inline std::string bestGuess(const std::vector<std::string>& candidates) {
    if (candidates.empty()) return "";

    std::array<std::array<int, 26>, 5> freq{};
    for (const auto& word : candidates) {
        for (int i = 0; i < 5; ++i) {
            freq[i][word[i] - 'a']++;
        }
    }

    auto scoreWord = [&](const std::string& word) {
        double score = 0.0;
        std::array<bool, 26> seenLetter{};
        for (int i = 0; i < 5; ++i) {
            int letter = word[i] - 'a';
            score += freq[i][letter];
            if (seenLetter[letter]) score *= 0.5; // repeats add less info
            seenLetter[letter] = true;
        }
        return score;
    };

    std::string best = candidates[0];
    double bestScore = -1.0;
    for (const auto& word : candidates) {
        double s = scoreWord(word);
        if (s > bestScore) {
            bestScore = s;
            best = word;
        }
    }
    return best;
}

} // namespace wordle

// ============================================================
// 3. CONSOLE
// ============================================================
// Everything below is console-specific I/O. It only ever talks to the
// wordle:: functions above through their public interface — it never
// touches how they're implemented. That's what makes them liftable into
// a shared header later without touching this section's logic.

namespace {

std::vector<std::string> loadWords(const std::string& path) {
    std::vector<std::string> words;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && !std::isalpha(static_cast<unsigned char>(line.back()))) {
            line.pop_back();
        }
        if (line.size() == 5) {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);
            words.push_back(line);
        }
    }
    return words;
}

wordle::Feedback parseFeedback(const std::string& input) {
    wordle::Feedback fb{};
    for (int i = 0; i < 5; ++i) {
        switch (std::tolower(static_cast<unsigned char>(input[i]))) {
            case 'g': fb[i] = wordle::Color::Green;  break;
            case 'y': fb[i] = wordle::Color::Yellow; break;
            default:  fb[i] = wordle::Color::Gray;   break;
        }
    }
    return fb;
}

} // namespace

int main() {
    auto words = loadWords("words.txt");
    if (words.empty()) {
        std::cerr << "Could not load words.txt (make sure it's in the same folder as the executable)\n";
        return 1;
    }

    std::vector<std::string> candidates = words;

    std::cout << "=== Wordle Solver ===\n";
    std::cout << "Play the real Wordle in another window. After each guess, tell me\n";
    std::cout << "what you guessed and the feedback colors (G=green, Y=yellow, B=gray).\n";
    std::cout << "Example: guessed 'crane', got gray/gray/green/yellow/gray -> type BBGYB\n\n";

    std::cout << "Loaded " << candidates.size() << " candidate words.\n";
    std::cout << "Suggested first guess: " << wordle::bestGuess(candidates) << "\n\n";

    while (candidates.size() > 1) {
        std::string guess, feedbackStr;

        std::cout << "Word you guessed: ";
        if (!(std::cin >> guess)) break;
        std::cout << "Feedback (5 letters G/Y/B): ";
        if (!(std::cin >> feedbackStr)) break;

        std::transform(guess.begin(), guess.end(), guess.begin(), ::tolower);
        if (guess.size() != 5 || feedbackStr.size() != 5) {
            std::cout << "Please enter exactly 5 characters for both.\n\n";
            continue;
        }

        wordle::Feedback fb = parseFeedback(feedbackStr);
        candidates = wordle::filterCandidates(candidates, guess, fb);

        std::cout << candidates.size() << " word(s) still possible.\n";
        if (candidates.size() <= 10 && !candidates.empty()) {
            std::cout << "  ";
            for (const auto& w : candidates) std::cout << w << " ";
            std::cout << "\n";
        }

        if (candidates.empty()) {
            std::cout << "No candidates left. The answer might not be in words.txt.\n";
            break;
        }
        if (candidates.size() > 1) {
            std::cout << "Suggested next guess: " << wordle::bestGuess(candidates) << "\n\n";
        }
    }

    if (candidates.size() == 1) {
        std::cout << "Solved! The word is: " << candidates.front() << "\n";
    }

    return 0;
}
