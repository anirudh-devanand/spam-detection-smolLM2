#!/usr/bin/env python3

import argparse
import os
import random
import re

import pandas as pd


SPAM_PHRASES = [
    "LIMITED TIME OFFER",
    "WIN BIG NOW",
    "CLICK HERE TO CLAIM YOUR PRIZE",
    "CONGRATULATIONS, YOU HAVE BEEN SELECTED",
    "Limited time offer just for you",
    "Win big now with this exclusive deal",
    "Click here to claim your prize today",
    "No credit check required for approval",
    "Exclusive deal just for you",
    "You have been selected for a special offer",
    "Act now before it's too late",
    "Earn extra money from home",
    "Make money while you sleep",
    "Guaranteed profits with minimal risk",
    "You won't believe this offer",
    "Increase your credit score fast",
    "Get rich quick with this simple trick",
    "No experience necessary, start today",
    "You have an unclaimed reward waiting",
    "Your account will be closed soon unless you act",
    "Final warning: update your details",
    "Your package is currently on hold",
    "Click to verify your delivery information",
    "You are eligible for a low-interest loan",
    "Approved in minutes with no documents required",
    "Exclusive access to a VIP investment",
    "Double your crypto balance in 24 hours",
    "Crypto opportunity you shouldn't miss",
    "Invest now and see huge returns",
    "You have been pre-approved for financing",
    "Your loan application is guaranteed to be accepted",
    "Claim your cashback reward today",
    "You may qualify for a special government grant",
    "Win an all-inclusive vacation for two",
    "Free gift card available for a limited time",
    "Get a brand new phone for a very low price",
    "Unlock your free subscription now",
    "Your password will expire soon",
    "Secure your account immediately by logging in",
    "Unusual sign-in activity detected on your account",
    "Activate your bonus before it expires",
    "Don't miss this once-in-a-lifetime deal",
    "Risk-free trial with no obligation",
    "Start earning instant commissions today",
    "Turn your spare time into extra cash",
    "Be your own boss starting today",
    "Claim your limited bonus credits now",
    "Exclusive offer for selected users only",
    "You are a guaranteed winner, check your numbers",
    "This message is not spam, please read",
    "Open immediately for important information",
]

HAM_PHRASES = [
    "Please let me know if you have any questions.",
    "Let me know what you think.",
    "Looking forward to your reply.",
    "Thanks and best regards.",
    "Happy to discuss this further.",
    "Hope you're doing well.",
    "Let me know if that works for you.",
    "Please confirm if this timeline is acceptable.",
    "Thanks again for your help with this.",
    "Feel free to reach out if anything is unclear.",
    "Please review and share your feedback.",
    "Let me know if you need any additional details.",
    "Looking forward to hearing your thoughts.",
    "Thanks for your time and consideration.",
    "Please see the attached document for more information.",
    "Let me know if the attached looks okay.",
    "Can we schedule a quick call to discuss?",
    "Please let me know what your availability looks like.",
    "Thanks for the quick turnaround on this.",
    "Appreciate your patience and support.",
    "Just checking in to see if you had a chance to look at this.",
    "Following up on my previous email.",
    "No rush, just wanted to keep this on your radar.",
    "Let me know if you'd like to go over this together.",
    "Thanks for bringing this to my attention.",
    "Please let me know if there are any changes.",
    "I'm happy to clarify anything if needed.",
    "Let me know if this aligns with your expectations.",
    "Looking forward to working with you on this.",
    "Thanks for the update.",
    "Please let me know if the new time works.",
    "Let’s plan to revisit this later this week.",
    "Thanks again and have a great day.",
    "Appreciate your feedback on this.",
    "Please let me know if you have any concerns.",
    "Let me know if you prefer another option.",
    "Thanks for your flexibility on this.",
    "I’ll wait for your confirmation before proceeding.",
    "Feel free to suggest a different approach.",
    "Please let me know if everything looks good.",
    "Looking forward to our meeting.",
    "Thanks for sharing these details.",
    "Let me know if you'd like me to make any changes.",
    "Happy to jump on a call to go through this.",
    "Thanks again for your collaboration.",
    "Please let me know once you’ve had a chance to review.",
    "I appreciate your prompt response.",
    "Let me know if you run into any issues.",
    "Thanks and talk soon.",
    "Hope you have a great rest of your day.",
]


def random_uppercase(s: str, p: float = 0.2) -> str:
    def transform_word(w):
        if random.random() < p:
            return w.upper()
        return w
    return " ".join(transform_word(w) for w in s.split())


def add_random_phrase(text: str, phrases, position: str = "end") -> str:
    phrase = random.choice(phrases)
    if position == "start":
        return phrase + " " + text
    elif position == "middle":
        parts = text.split()
        if len(parts) < 2:
            return text + " " + phrase
        i = random.randint(1, len(parts) - 1)
        return " ".join(parts[:i] + [phrase] + parts[i:])
    else:
        return text + " " + phrase


def perturb_punctuation(text: str) -> str:
    text = re.sub(r"!+", "!", text)
    if random.random() < 0.5:
        text = text + "!"
    if random.random() < 0.3:
        text = text.replace(".", "!!")
    return text


def augment_spam(subject: str, message: str):
    subject_new = random_uppercase(subject, p=0.4)
    subject_new = add_random_phrase(subject_new, SPAM_PHRASES, position=random.choice(["start", "end"]))
    message_new = add_random_phrase(message, SPAM_PHRASES, position=random.choice(["middle", "end"]))
    message_new = perturb_punctuation(message_new)
    return subject_new, message_new


def augment_ham(subject: str, message: str):
    if random.random() < 0.3:
        subject_new = subject + " (follow up)"
    else:
        subject_new = subject
    message_new = message
    if random.random() < 0.7:
        message_new = add_random_phrase(message_new, HAM_PHRASES, position="end")
    return subject_new, message_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default="autograder/cpen455_released_datasets/train_val_subset.csv",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="autograder/cpen455_released_datasets/train_val_augmented.csv",
    )
    parser.add_argument("--spam_factor", type=int, default=1)
    parser.add_argument("--ham_factor", type=int, default=1)
    args = parser.parse_args()

    assert os.path.exists(args.input_csv), f"{args.input_csv} not found"

    df = pd.read_csv(args.input_csv)

    cols = [c.lower() for c in df.columns]
    assert "subject" in cols and "message" in cols and ("spam/ham" in cols or "label" in cols), \
        f"Expected columns like Subject, Message, Spam/Ham, got {df.columns}"

    col_map = {c: c for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        if cl == "subject":
            col_map[c] = "Subject"
        elif cl == "message":
            col_map[c] = "Message"
        elif cl in ("spam/ham", "label"):
            col_map[c] = "Spam/Ham"

    df = df.rename(columns=col_map)

    if "Index" not in df.columns:
        df.insert(0, "Index", range(len(df)))

    synthetic_rows = []
    for _, row in df.iterrows():
        subj = str(row["Subject"])
        msg = str(row["Message"])
        raw = row["Spam/Ham"]

        try:
            num = int(raw)
        except (ValueError, TypeError):
            continue  # skip bad labels

        if num == 1:
            k = args.spam_factor
            for _ in range(k):
                s_new, m_new = augment_spam(subj, msg)
                synthetic_rows.append(
                    {
                        "Index": -1,
                        "Subject": s_new,
                        "Message": m_new,
                        "Spam/Ham": 1,
                    }
                )
        elif num == 0:
            k = args.ham_factor
            for _ in range(k):
                s_new, m_new = augment_ham(subj, msg)
                synthetic_rows.append(
                    {
                        "Index": -1,
                        "Subject": s_new,
                        "Message": m_new,
                        "Spam/Ham": 0,
                    }
                )
        else:
            continue

    df_synth = pd.DataFrame(synthetic_rows)
    df_out = pd.concat([df, df_synth], ignore_index=True)

    df_out["Index"] = range(len(df_out))

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(
        f"Saved augmented dataset to {args.output_csv} "
        f"with {len(df_out)} rows (original {len(df)} + synthetic {len(df_synth)})"
    )
