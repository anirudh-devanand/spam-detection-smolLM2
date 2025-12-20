PROMPT_TEMPLATE = (
    "{user_prompt}\n"
    "The following email is labeled as {label}.\n"
    "Subject: {subject}\n"
    "Message: {message}"
)

def get_prompt(subject: str, message: str, label: str, max_seq_length: int = 256, user_prompt: str = "") -> str:
    prompt = PROMPT_TEMPLATE.format(user_prompt=user_prompt, subject=subject, message=message, label=label)
    prompt = prompt[:max_seq_length]
    return prompt

if __name__ == "__main__":

    print(get_prompt(
        user_prompt="Please classify the email below as ham or spam.",
        subject="Hello", 
        message="This is a test message.", 
        label="ham", 
        max_seq_length=256))
