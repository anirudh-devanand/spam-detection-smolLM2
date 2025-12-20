uv run -m examples.bayes_inverse \
--method naive_prompting \
--max_seq_len 512 \
--batch_size 1 \
--user_prompt "You are an email classifier for a corporate dataset.\nOutput exactly one lowercase token: spam or ham.\nHam: direct, personal/operational notes (replies, questions, scheduling, task-specific), short and focused, addressed to a specific person/team.\nSpam if either:\nA) Automated/system notice: repetitive subject/body; log/error text (e.g., ‘general sql error’, ‘parsing file’, ‘start date : … ; hourahead hour : … ;’).\nB) Impersonal memo: very long, broad distribution/forward chain; corporate/legal/financial content; contains a long boilerplate legal disclaimer/footer.\nRule: If any spam cue matches → spam; else → ham.\nReturn only the label with no explanation."
