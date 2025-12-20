# This config spends about 3 minutes to run on a Macmini with M4 Chip, 16GB RAM
uv run -m examples.bayes_inverse \
--method full_finetune \
--max_seq_len 256 \
--batch_size 8 \
--num_iterations 80