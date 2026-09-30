// STAGE 6 of 6 — Entropy strategy, colorized console, cached opening guess
//
// Everything from stage 5 is still here unchanged (engine, frequency
// strategy, console loop). Three things are new:
//   1. bestGuessEntropy() — a strategy that scores guesses by expected
//      information gain (Shannon entropy) instead of raw letter frequency.
//   2. cachedOpeningGuess() — the first guess never changes for a given
//      word list, so it's computed once and cached to disk.
//   3. printColoredGuess() — feedback prints as colored tiles instead of
//      a "BBGYB" string.

#include <array>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace wordle {

// ============================================================
// ENGINE — unchanged since stage 3
// ============================================================

enum class Color { Gray, Yellow, Green };
using Feedback = std::array<Color, 5>;

Feedback computeFeedback(const std::string& guess, const std::string& answer) {
    Feedback result{};
    result.fill(Color::Gray);

    std::array<bool, 5> answerLetterUsed{};
    answerLetterUsed.fill(false);

    for (int i = 0; i < 5; ++i) {
        if (guess[i] == answer[i]) {
            result[i] = Color::Green;
            answerLetterUsed[i] = true;
        }
    }

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

bool isConsistent(const std::string& candidate, const std::string& guess,
                   const Feedback& feedback) {
    return computeFeedback(guess, candidate) == feedback;
}

std::vector<std::string> filterCandidates(const std::vector<std::string>& candidates,
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

// ============================================================
// STRATEGY (frequency) — unchanged since stage 4, kept around so you can
// still compare it against the entropy strategy below if you want.
// ============================================================

std::string bestGuess(const std::vector<std::string>& candidates) {
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

// ============================================================
// STRATEGY (entropy) — NEW in stage 6
// ============================================================

// Encodes a feedback pattern as a single integer in [0, 243) — base 3 over
// 5 tiles (Gray=0, Yellow=1, Green=2). Lets us bucket feedback patterns in
// a plain fixed-size array instead of hashing a Feedback struct.
int patternIndex(const Feedback& fb) {
    int idx = 0;
    for (int i = 0; i < 5; ++i) {
        idx = idx * 3 + static_cast<int>(fb[i]);
    }
    return idx;
}

// Same computation as computeFeedback, but returns the packed integer
// directly. This is the innermost loop of the entropy search (it runs
// candidates x candidates times for the opening guess), so skipping the
// intermediate Feedback array here is worth the code duplication.
int patternIndexOf(const std::string& guess, const std::string& answer) {
    std::array<int, 5> tile{};
    std::array<bool, 5> answerLetterUsed{};
    answerLetterUsed.fill(false);

    for (int i = 0; i < 5; ++i) {
        if (guess[i] == answer[i]) {
            tile[i] = 2; // Green
            answerLetterUsed[i] = true;
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (tile[i] == 2) continue;
        tile[i] = 0; // Gray unless proven otherwise below
        for (int j = 0; j < 5; ++j) {
            if (!answerLetterUsed[j] && guess[i] == answer[j]) {
                tile[i] = 1; // Yellow
                answerLetterUsed[j] = true;
                break;
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < 5; ++i) idx = idx * 3 + tile[i];
    return idx;
}

constexpr int kPatternCount = 243; // 3^5

// Scores a candidate guess by expected information gain: split
// `possibleAnswers` into buckets by the feedback pattern this guess would
// produce against each one, then take the Shannon entropy (in bits) of
// that bucket-size distribution. A guess that spreads answers evenly
// across many buckets narrows the field faster than one that mostly comes
// back all-gray, even if the all-gray guess "sounds" more informative.
double entropyOf(const std::string& guess, const std::vector<std::string>& possibleAnswers) {
    std::array<int, kPatternCount> buckets{};
    for (const auto& answer : possibleAnswers) {
        buckets[patternIndexOf(guess, answer)]++;
    }

    double total = static_cast<double>(possibleAnswers.size());
    double entropy = 0.0;
    for (int count : buckets) {
        if (count == 0) continue;
        double p = count / total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

// Picks the guess (from `guessPool`) with the highest expected information
// gain against `possibleAnswers`. `guessPool` doesn't have to be the same
// as `possibleAnswers` — real Wordle solvers often gain more by guessing a
// word that can't itself be the answer, purely because it splits the
// remaining candidates better than any actual candidate would. Ties are
// broken in favor of a guess that could itself be the answer, since a
// correct guess ends the game outright even when it splits the field no
// better than an alternative.
std::string bestGuessEntropy(const std::vector<std::string>& possibleAnswers,
                              const std::vector<std::string>& guessPool) {
    if (possibleAnswers.empty()) return "";
    if (possibleAnswers.size() == 1) return possibleAnswers.front();

    auto isCandidate = [&](const std::string& w) {
        return std::find(possibleAnswers.begin(), possibleAnswers.end(), w) != possibleAnswers.end();
    };

    std::string best;
    double bestScore = -1.0;
    bool bestIsCandidate = false;

    for (const auto& guess : guessPool) {
        double score = entropyOf(guess, possibleAnswers);
        bool candidate = isCandidate(guess);

        bool better = score > bestScore + 1e-9 ||
                      (std::abs(score - bestScore) <= 1e-9 && candidate && !bestIsCandidate);

        if (better) {
            bestScore = score;
            best = guess;
            bestIsCandidate = candidate;
        }
    }
    return best;
}

// Convenience overload for callers that don't maintain a separate, larger
// "valid guesses" list — just search the answer pool itself.
std::string bestGuessEntropy(const std::vector<std::string>& possibleAnswers) {
    return bestGuessEntropy(possibleAnswers, possibleAnswers);
}

// ============================================================
// OPENING-GUESS CACHE — NEW in stage 6
// ============================================================
//
// The first guess is a pure function of words.txt's contents, so it's
// always the same until the word list changes. Finding it via entropy
// search means scoring every word against every other word (O(n^2)),
// which can take a noticeable moment on a large list. We cache it to disk
// keyed by a hash of the list, so every run after the first gets it back
// instantly instead of recomputing it.

namespace detail {

// Small FNV-1a hash over the concatenated word list. Only used to detect
// "is this the same words.txt as last time" — not security sensitive.
uint64_t hashWords(const std::vector<std::string>& words) {
    uint64_t h = 1469598103934665603ull; // FNV offset basis
    auto mix = [&](unsigned char c) {
        h ^= c;
        h *= 1099511628211ull; // FNV prime
    };
    for (const auto& w : words) {
        for (unsigned char c : w) mix(c);
        mix('\n');
    }
    return h;
}

} // namespace detail

// Returns the best opening guess for `words`, using a cache file at
// `cachePath` to skip the entropy search on subsequent runs. Automatically
// recomputes if the word list has changed since the cache was written.
std::string cachedOpeningGuess(const std::vector<std::string>& words,
                                const std::string& cachePath = "opening_guess_cache.txt") {
    uint64_t hash = detail::hashWords(words);

    std::ifstream in(cachePath);
    if (in) {
        uint64_t cachedHash = 0;
        std::string cachedGuess;
        if (in >> cachedHash >> cachedGuess && cachedHash == hash && cachedGuess.size() == 5) {
            return cachedGuess;
        }
    }

    std::string guess = bestGuessEntropy(words);

    std::ofstream out(cachePath, std::ios::trunc);
    if (out) {
        out << hash << " " << guess << "\n";
    }
    return guess;
}

} // namespace wordle

// ============================================================
// CONSOLE — colorized output, new in stage 6
// ============================================================

namespace {

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

// ANSI escape codes for colored tile backgrounds. On a terminal that
// doesn't support color these just print as a few stray characters around
// otherwise-readable text, so there's no hard dependency on support.
constexpr const char* kReset  = "\033[0m";
constexpr const char* kGreen  = "\033[1;97;42m";  // white bold on green
constexpr const char* kYellow = "\033[1;97;43m";  // white bold on yellow
constexpr const char* kGray   = "\033[1;97;100m"; // white bold on gray

const char* colorFor(wordle::Color c) {
    switch (c) {
        case wordle::Color::Green:  return kGreen;
        case wordle::Color::Yellow: return kYellow;
        default:                    return kGray;
    }
}

// Prints the guessed word as colored tiles, e.g. a green-backed "C" for a
// correct letter in the right spot, so the feedback you just typed is easy
// to double-check at a glance instead of re-reading a "BBGYB" string.
void printColoredGuess(const std::string& guess, const wordle::Feedback& fb) {
    for (int i = 0; i < 5; ++i) {
        char letter = static_cast<char>(std::toupper(static_cast<unsigned char>(guess[i])));
        std::cout << colorFor(fb[i]) << ' ' << letter << ' ' << kReset;
    }
    std::cout << "\n";
}

} // namespace

int main() {
    auto words = wordle::loadWords("words.txt");
    if (words.empty()) {
        std::cerr << "Could not load words.txt (make sure it's in the same folder as the executable)\n";
        return 1;
    }

    std::vector<std::string> candidates = words;

    std::cout << "=== Wordle Solver (stage 6: entropy strategy) ===\n";
    std::cout << "Play the real Wordle in another window. After each guess, tell me\n";
    std::cout << "what you guessed and the feedback colors (G=green, Y=yellow, B=gray).\n";
    std::cout << "Example: guessed 'crane', got gray/gray/green/yellow/gray -> type BBGYB\n\n";

    std::cout << "Loaded " << candidates.size() << " candidate words.\n";

    auto t0 = std::chrono::steady_clock::now();
    std::string opening = wordle::cachedOpeningGuess(candidates);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Suggested first guess: " << opening
               << "  (" << ms << " ms — cached after the first run)\n\n";

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
        printColoredGuess(guess, fb);

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
            // Search the full word list, not just the shrunken candidate
            // pool — a non-candidate "probe" word sometimes splits the
            // field better than any word that could actually be the answer.
            std::cout << "Suggested next guess: "
                      << wordle::bestGuessEntropy(candidates, words) << "\n\n";
        }
    }

    if (candidates.size() == 1) {
        std::cout << kGreen << " Solved! The word is: " << candidates.front() << " " << kReset << "\n";
    }

    return 0;
}
