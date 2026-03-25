"""
generate_dataset.py
Entropic ASR Pipeline — Synthetic Financial Benchmark Generator

Generates 300 linguistically diverse Hinglish financial utterances
designed for efficient model learning:
  - Varied code-switching density (heavy Hindi, heavy English, balanced)
  - Amount in different sentence positions (start, middle, end)
  - Different registers (casual, formal, urgent)
  - Disambiguation cases for all 5 intents
  - Full number range coverage (1 to 10 lakh)

Output: data/raw/synthetic_benchmark.csv
"""

import csv
import random
import os
from itertools import product

random.seed(42)  # reproducible

# ── NUMBER SYSTEM ──────────────────────────────────────────────────────────────
# Each entry: (spoken_hinglish, normalized_int, display_str)
NUMBERS = {
    "ones": [
        ("ek",          1,        "₹1"),
        ("do",          2,        "₹2"),       # AMBIGUOUS — also verb
        ("teen",        3,        "₹3"),
        ("chaar",       4,        "₹4"),
        ("paanch",      5,        "₹5"),
        ("chheh",       6,        "₹6"),
        ("saat",        7,        "₹7"),
        ("aath",        8,        "₹8"),
        ("nau",         9,        "₹9"),
    ],
    "tens": [
        ("das",         10,       "₹10"),
        ("bees",        20,       "₹20"),
        ("tees",        30,       "₹30"),
        ("pachaas",     50,       "₹50"),
        ("pachattar",   75,       "₹75"),
    ],
    "hundreds": [
        ("sau",         100,      "₹100"),
        ("do sau",      200,      "₹200"),      # AMBIGUOUS
        ("paanch sau",  500,      "₹500"),
        ("aath sau",    800,      "₹800"),
    ],
    "thousands": [
        ("ek hazaar",       1000,     "₹1,000"),
        ("do hazaar",       2000,     "₹2,000"),  # AMBIGUOUS
        ("paanch hazaar",   5000,     "₹5,000"),
        ("das hazaar",      10000,    "₹10,000"),
        ("pachees hazaar",  25000,    "₹25,000"),
    ],
    "compound": [
        ("do hazaar paanch sau",        2500,     "₹2,500"),  # AMBIGUOUS
        ("ek hazaar do sau",            1200,     "₹1,200"),
        ("teen hazaar paanch sau",      3500,     "₹3,500"),
        ("paanch hazaar do sau",        5200,     "₹5,200"),
        ("das hazaar paanch sau",       10500,    "₹10,500"),
    ],
    "lakhs": [
        ("ek lakh",             100000,   "₹1,00,000"),
        ("do lakh",             200000,   "₹2,00,000"),   # AMBIGUOUS
        ("paanch lakh",         500000,   "₹5,00,000"),
        ("ek lakh paanch hazaar", 105000, "₹1,05,000"),
        ("do lakh teen hazaar",   203000, "₹2,03,000"),   # AMBIGUOUS
    ],
}

ALL_NUMBERS = []
for group in NUMBERS.values():
    ALL_NUMBERS.extend(group)

# ── DISAMBIGUATION CASES ───────────────────────────────────────────────────────
# "do" appears but is a VERB not a number
# Format: (transcript, normalized, intent, amount_int, note)
DISAMBIGUATION_CASES = {
    "SEND_MONEY": [
        ("do na yaar transfer abhi",
         "Please do the transfer now",
         None, "do=VERB:imperative"),

        ("bhai do de usse paisa",
         "Bro give him the money",
         None, "do=VERB:give"),

        ("do diya kya tune transaction",
         "Did you complete the transaction",
         None, "do=VERB:past"),

        ("yaar ek baar do kar transfer",
         "Bro do the transfer once",
         None, "do=VERB:imperative"),

        ("please do it abhi bhej",
         "Please send it now",
         None, "do=ENGLISH_VERB"),
    ],
    "RECEIVE_MONEY": [
        ("do diya kya usne paisa",
         "Did he give the money",
         None, "do=VERB:gave"),

        ("usne do nahi kiya abhi tak",
         "He hasn't done it yet",
         None, "do=VERB:negation"),
    ],
    "CHECK_BALANCE": [
        ("do baar check kiya balance",
         "Checked balance twice",
         None, "do=ADVERB:twice_not_number"),

        ("ek do minute mein dekh balance",
         "Check balance in a minute or two",
         None, "do=FILLER:approximate"),
    ],
    "EXPENSE_LOG": [
        ("aaj do baar kharcha hua",
         "Spent twice today",
         None, "do=ADVERB:frequency"),

        ("do teen cheezein kharidi aaj",
         "Bought two or three things today",
         None, "do=APPROXIMATION"),
    ],
    "BILL_PAYMENT": [
        ("bijli ka bill do na yaar",
         "Please pay the electricity bill",
         None, "do=VERB:imperative"),

        ("do diya kya bill",
         "Was the bill paid",
         None, "do=VERB:past"),
    ],
}

# ── SLOT POOLS ─────────────────────────────────────────────────────────────────

SEND_TEMPLATES = [
    # (template_fn, register)
    # template_fn(spoken_amt, display_amt) -> (transcript, normalized)

    # CASUAL
    (lambda s, d: (f"{s} bhejo usse",                   f"Send {d}"),                     "casual"),
    (lambda s, d: (f"bhai {s} bhej de",                 f"Send {d}"),                     "casual"),
    (lambda s, d: (f"yaar {s} transfer kar",            f"Transfer {d}"),                 "casual"),
    (lambda s, d: (f"{s} bhijwa de usse",               f"Send {d} to him"),              "casual"),
    (lambda s, d: (f"abhi {s} bhejo",                   f"Send {d} now"),                 "urgent"),
    (lambda s, d: (f"jaldi {s} transfer kar yaar",      f"Transfer {d} quickly"),         "urgent"),
    (lambda s, d: (f"{s} de do usse please",            f"Please give {d}"),              "casual"),
    (lambda s, d: (f"ek baar {s} bhej",                 f"Send {d} once"),                "casual"),

    # FORMAL
    (lambda s, d: (f"mujhe {s} transfer karne hain",    f"I need to transfer {d}"),       "formal"),
    (lambda s, d: (f"{s} ka payment karna hai",         f"Need to make payment of {d}"),  "formal"),
    (lambda s, d: (f"account mein {s} bhejne hain",     f"Need to send {d} to account"),  "formal"),

    # AMOUNT AT END
    (lambda s, d: (f"usse bhejo {s}",                   f"Send {d} to him"),              "casual"),
    (lambda s, d: (f"transfer kar de {s} abhi",         f"Transfer {d} now"),             "urgent"),
    (lambda s, d: (f"phonepe pe bhejo {s}",             f"Send {d} on PhonePe"),          "casual"),
    (lambda s, d: (f"gpay kar do {s}",                  f"GPay {d}"),                     "casual"),

    # CONTEXTUAL
    (lambda s, d: (f"bhai rent ke liye {s} chahiye",    f"Need {d} for rent"),            "casual"),
    (lambda s, d: (f"mummy ko {s} bhejne hain",         f"Send {d} to mummy"),            "casual"),
    (lambda s, d: (f"dost ko {s} wapas karne hain",     f"Return {d} to friend"),         "casual"),
    (lambda s, d: (f"uska {s} baaki hai bhej",          f"Send remaining {d}"),           "casual"),
    (lambda s, d: (f"{s} ka transaction karo abhi",     f"Do transaction of {d} now"),    "urgent"),
]

RECEIVE_TEMPLATES = [
    (lambda s, d: (f"{s} aaya kya mere account mein",   f"Did {d} arrive in my account"), "casual"),
    (lambda s, d: (f"mera {s} nahi aaya yaar",          f"My {d} hasn't arrived"),        "casual"),
    (lambda s, d: (f"{s} receive hua kya",              f"Was {d} received"),             "casual"),
    (lambda s, d: (f"usne {s} bheja kya",               f"Did he send {d}"),              "casual"),
    (lambda s, d: (f"{s} pending hai abhi tak",         f"{d} is still pending"),         "casual"),
    (lambda s, d: (f"bhai {s} kab aayega",              f"When will {d} arrive bro"),     "casual"),
    (lambda s, d: (f"mujhe {s} receive karna hai",      f"I need to receive {d}"),        "formal"),
    (lambda s, d: (f"{s} ka transaction complete hua kya", f"Did {d} transaction complete"), "formal"),
    (lambda s, d: (f"account mein {s} dikha raha hai kya", f"Is {d} showing in account"), "formal"),
    (lambda s, d: (f"yaar {s} aate aate teen din ho gaye", f"Three days waiting for {d}"), "casual"),
]

BALANCE_TEMPLATES = [
    (lambda s, d: (f"balance check karo",               f"Check balance"),                "casual"),
    (lambda s, d: (f"mera balance kitna hai",           f"What is my balance"),           "casual"),
    (lambda s, d: (f"account mein kitne paise hain",    f"How much money in account"),    "casual"),
    (lambda s, d: (f"balance dikha",                    f"Show balance"),                 "casual"),
    (lambda s, d: (f"abhi balance kya hai",             f"What is current balance"),      "casual"),
    (lambda s, d: (f"balance statement chahiye",        f"Need balance statement"),       "formal"),
    (lambda s, d: (f"kitna balance bacha hai",          f"How much balance remaining"),   "casual"),
    (lambda s, d: (f"account balance check karna tha",  f"Wanted to check account balance"), "formal"),
    (lambda s, d: (f"bhai balance low ho gaya kya",     f"Has balance gone low bro"),     "casual"),
    (lambda s, d: (f"available balance kitna hai",      f"How much available balance"),   "formal"),
]

EXPENSE_TEMPLATES = [
    (lambda s, d: (f"aaj {s} ka kharcha hua",           f"Spent {d} today"),              "casual"),
    (lambda s, d: (f"{s} ka samaan liya aaj",           f"Bought goods worth {d} today"), "casual"),
    (lambda s, d: (f"subah {s} kharcha kiya",           f"Spent {d} in morning"),         "casual"),
    (lambda s, d: (f"{s} gaya petrol mein",             f"{d} spent on petrol"),          "casual"),
    (lambda s, d: (f"dukan ka kiraya {s} diya",         f"Paid shop rent {d}"),           "casual"),
    (lambda s, d: (f"{s} ka wholesale maal liya",       f"Bought wholesale stock of {d}"), "casual"),
    (lambda s, d: (f"aaj ka total kharcha {s} hua",     f"Total expense today {d}"),      "formal"),
    (lambda s, d: (f"{s} kharch ho gaya yaar",          f"{d} got spent"),                "casual"),
    (lambda s, d: (f"khana kharid liya {s} ka",         f"Bought food worth {d}"),        "casual"),
    (lambda s, d: (f"bijli ka bill {s} aaya",           f"Electricity bill came to {d}"), "casual"),
    (lambda s, d: (f"{s} ki dawai leni thi",            f"Had to buy medicine worth {d}"), "casual"),
    (lambda s, d: (f"auto ka kiraya {s} laga",          f"Auto fare was {d}"),            "casual"),
]

BILL_TEMPLATES = [
    (lambda s, d: (f"bijli ka bill bharo {s}",          f"Pay electricity bill {d}"),     "casual"),
    (lambda s, d: (f"{s} ka recharge karna hai",        f"Need to recharge for {d}"),     "casual"),
    (lambda s, d: (f"DTH recharge karo {s} ka",        f"Do DTH recharge of {d}"),       "casual"),
    (lambda s, d: (f"{s} ka gas bill aaya hai",         f"Gas bill of {d} has arrived"),  "casual"),
    (lambda s, d: (f"internet bill {s} dena hai",       f"Need to pay internet bill {d}"), "formal"),
    (lambda s, d: (f"{s} ka mobile recharge chahiye",   f"Need mobile recharge of {d}"),  "casual"),
    (lambda s, d: (f"pani ka bill {s} tha",             f"Water bill was {d}"),           "casual"),
    (lambda s, d: (f"{s} ka postpaid bill pay karo",    f"Pay postpaid bill of {d}"),     "formal"),
    (lambda s, d: (f"bhai {s} ka recharge kar de",      f"Bro do recharge of {d}"),       "casual"),
    (lambda s, d: (f"{s} mein bill settle ho jayega",   f"Bill of {d} will be settled"),  "formal"),
]

INTENT_CONFIG = {
    "SEND_MONEY":    {"templates": SEND_TEMPLATES,    "count": 65, "needs_amount": True},
    "RECEIVE_MONEY": {"templates": RECEIVE_TEMPLATES, "count": 40, "needs_amount": True},
    "CHECK_BALANCE": {"templates": BALANCE_TEMPLATES, "count": 40, "needs_amount": False},
    "EXPENSE_LOG":   {"templates": EXPENSE_TEMPLATES, "count": 50, "needs_amount": True},
    "BILL_PAYMENT":  {"templates": BILL_TEMPLATES,    "count": 45, "needs_amount": True},
}
# 65+40+40+50+45 = 240 template-based + 20 disambiguation = 260 base
# We'll pad to 300 with extra variation below


def pick_numbers(n, avoid_do_ambiguous=False):
    """Pick n distinct number entries spread across all ranges."""
    pool = ALL_NUMBERS.copy()
    if avoid_do_ambiguous:
        pool = [(s, v, d) for s, v, d in pool if not s.startswith("do")]
    random.shuffle(pool)
    selected = []
    seen_values = set()
    for entry in pool:
        if entry[1] not in seen_values:
            selected.append(entry)
            seen_values.add(entry[1])
        if len(selected) >= n:
            break
    # if not enough unique, allow repeats
    while len(selected) < n:
        selected.append(random.choice(pool))
    return selected


def generate_template_rows(intent, templates, count, needs_amount):
    rows = []
    numbers = pick_numbers(count * 2)  # oversample, then trim
    num_idx = 0

    for i in range(count):
        tmpl_fn, register = templates[i % len(templates)]

        if needs_amount:
            spoken_amt, amount_int, display_amt = numbers[num_idx % len(numbers)]
            num_idx += 1
            transcript, normalized = tmpl_fn(spoken_amt, display_amt)
        else:
            transcript, normalized = tmpl_fn("", "")
            amount_int = None

        rows.append({
            "audio_path":   "",
            "transcript":   transcript.strip(),
            "normalized":   normalized.strip(),
            "intent":       intent,
            "amount":       amount_int if amount_int else "",
            "is_disambiguation": "False",
            "register":     register,
            "note":         "",
        })
    return rows


def generate_disambiguation_rows():
    rows = []
    for intent, cases in DISAMBIGUATION_CASES.items():
        for transcript, normalized, amount, note in cases:
            rows.append({
                "audio_path":   "",
                "transcript":   transcript,
                "normalized":   normalized,
                "intent":       intent,
                "amount":       amount if amount else "",
                "is_disambiguation": "True",
                "register":     "casual",
                "note":         note,
            })
    return rows


def generate_extra_variation_rows(needed):
    """
    Generate extra rows by combining templates with number ranges
    not yet covered, ensuring variety.
    """
    rows = []
    intents_with_amounts = ["SEND_MONEY", "RECEIVE_MONEY", "EXPENSE_LOG", "BILL_PAYMENT"]

    # Cover all number ranges at least once per intent
    for intent in intents_with_amounts:
        cfg = INTENT_CONFIG[intent]
        for group_name, numbers in NUMBERS.items():
            for spoken_amt, amount_int, display_amt in numbers:
                if len(rows) >= needed:
                    break
                tmpl_fn, register = random.choice(cfg["templates"])
                transcript, normalized = tmpl_fn(spoken_amt, display_amt)
                rows.append({
                    "audio_path":   "",
                    "transcript":   transcript.strip(),
                    "normalized":   normalized.strip(),
                    "intent":       intent,
                    "amount":       amount_int,
                    "is_disambiguation": "False",
                    "register":     register,
                    "note":         f"range:{group_name}",
                })
            if len(rows) >= needed:
                break
        if len(rows) >= needed:
            break
    return rows[:needed]


def main():
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/synthetic_benchmark.csv"

    all_rows = []

    # Step 1: Template-based rows per intent
    print("Generating template-based rows...")
    for intent, cfg in INTENT_CONFIG.items():
        rows = generate_template_rows(
            intent,
            cfg["templates"],
            cfg["count"],
            cfg["needs_amount"]
        )
        all_rows.extend(rows)
        print(f"  {intent}: {len(rows)} rows")

    # Step 2: Disambiguation cases (all intents)
    print("Generating disambiguation cases...")
    disambig_rows = generate_disambiguation_rows()
    all_rows.extend(disambig_rows)
    print(f"  Disambiguation: {len(disambig_rows)} rows")

    # Step 3: Fill to 300
    current = len(all_rows)
    needed = 300 - current
    if needed > 0:
        print(f"Generating {needed} extra variation rows to reach 300...")
        extra = generate_extra_variation_rows(needed)
        all_rows.extend(extra)

    # Step 4: Shuffle (keep it from being ordered by intent)
    random.shuffle(all_rows)

    # Step 5: Write CSV
    fieldnames = [
        "audio_path", "transcript", "normalized",
        "intent", "amount", "is_disambiguation", "register", "note"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Total rows: {len(all_rows)}")
    print(f"Saved to: {output_path}")

    # Step 6: Print stats
    print("\n── Distribution ─────────────────────────")
    from collections import Counter
    intent_counts = Counter(r["intent"] for r in all_rows)
    for intent, count in sorted(intent_counts.items()):
        disambig = sum(1 for r in all_rows if r["intent"] == intent and r["is_disambiguation"] == "True")
        print(f"  {intent:<20} total: {count:>3}  |  disambiguation: {disambig}")

    print("\n── Register Distribution ─────────────────")
    register_counts = Counter(r["register"] for r in all_rows)
    for reg, count in sorted(register_counts.items()):
        print(f"  {reg:<10} {count}")

    print("\n── Number Range Coverage ─────────────────")
    range_labels = ["ones", "tens", "hundreds", "thousands", "compound", "lakhs"]
    for group_name, numbers in NUMBERS.items():
        spoken_set = {s for s, _, _ in numbers}
        hits = sum(1 for r in all_rows
                   if any(s in r["transcript"] for s in spoken_set)
                   and r["is_disambiguation"] == "False")
        print(f"  {group_name:<12} ~{hits} rows contain this range")

    print("\n── Sample Rows ───────────────────────────")
    samples = random.sample(all_rows, min(8, len(all_rows)))
    for s in samples:
        disambig_tag = " [DISAMBIG]" if s["is_disambiguation"] == "True" else ""
        print(f"  [{s['intent']}{disambig_tag}]")
        print(f"    IN:  {s['transcript']}")
        print(f"    OUT: {s['normalized']}  |  amount={s['amount']}")
        print()


if __name__ == "__main__":
    main()