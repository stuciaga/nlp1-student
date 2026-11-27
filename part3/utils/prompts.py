"""Binary IMDB prompt + robust parser (no torch dependency, unit-testable).

Labels follow Part 1: 0 = negative, 1 = positive.
Framing: multiple-choice letter. `order` controls which label gets letter A, so the
label-order permutation stress test (workplan 5.3.4) is just order=("positive","negative").
"""
import re

LABEL_WORDS = {0: "negative", 1: "positive"}
LETTERS = ["A", "B"]

# Instruction wordings for the --instruction_id ablation (workplan 2.1).
INSTRUCTIONS = {
    0: "You are a bot that predicts the sentiment of a movie review. The options are: {opts}. "
       "Just respond with the letter of the sentiment ({letters}).",
    1: "Classify the sentiment of the following movie review. Options: {opts}. "
       "Answer with a single letter ({letters}) and nothing else.",
    2: "Decide whether the reviewer liked or disliked the movie. Options: {opts}. "
       "Reply only with {letters}.",
    3: "You are a careful sentiment analyst. Read the review and choose the option that best "
       "describes its overall sentiment: {opts}. Output only the letter ({letters}).",
}


def letter_map(order=("negative", "positive")):
    """label id -> letter, given which label word comes first."""
    inv = {w: i for i, w in LABEL_WORDS.items()}
    return {inv[w]: LETTERS[k] for k, w in enumerate(order)}


def build_instruction(instruction_id=0, order=("negative", "positive")):
    lm = letter_map(order)
    opts = ", ".join(f"({lm[l]}) {LABEL_WORDS[l]}" for l in sorted(lm, key=lambda l: lm[l]))
    return INSTRUCTIONS[instruction_id].format(opts=opts, letters="/".join(LETTERS))


def format_demos(examples, order=("negative", "positive")):
    """examples: list of (text, label). Same 'Input:/Answer:' format as the user turn."""
    lm = letter_map(order)
    out = "Here are some examples of inputs and outputs:\n"
    for text, label in examples:
        out += f"Input: {text}\nAnswer: {lm[label]}\n"
    return out


_LETTER_RE = re.compile(r"^\W*(?:answer\W*)?\(?([AB])\)?(?![A-Za-z])", re.IGNORECASE)


def parse_generated_string(s, order=("negative", "positive")):
    """Return label id (0/1) or -1 if invalid.

    Accepts 'B', '(B)', 'Answer: B', 'B. positive'. Falls back to a lone label word
    ('positive'/'negative'). Ambiguous output (both/neither) is invalid, unlike the
    tutorial's `"A" in s` check, which turned 'Answer: B' into class 0.
    """
    inv_letter = {v: k for k, v in letter_map(order).items()}
    m = _LETTER_RE.match(s.strip())
    if m:
        return inv_letter[m.group(1).upper()]
    low = s.lower()
    has = [i for i, w in LABEL_WORDS.items() if re.search(rf"\b{w}\b", low)]
    return has[0] if len(has) == 1 else -1


def invalid_rate(labels):
    return sum(l == -1 for l in labels) / max(len(labels), 1)
