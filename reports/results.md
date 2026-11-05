# unslop: results

## Verdict

- **NO CLAIM**: after length control, the difference is not significant (LC win rate 49.1% [27.3%, 74.5%]).
- **WARN (Goodhart gap)**: the judge used for training rates the tuned model +20% higher than the independent judge. Part of the gain is fitting one judge's taste.
- No human eval labels yet, so every claim above rests on LLM judges.

## Tuned vs base, pairwise (95% bootstrap CI)

| Judge | Model | n judged / attempted | Win rate | Length-controlled win rate | Position-consistent | Tuned longer |
|---|---|---|---|---|---|---|
| eval | `groq/openai/gpt-oss-120b` | 16/16 | 48.4% [28.1%, 70.3%] | 49.1% [27.3%, 74.5%] | 69% | 44% |
| train | `gemini/gemini-3.8-flash` | 16/16 | 68.8% [53.1%, 84.4%] | 68.8% [49.3%, 86.7%] | 88% | 44% |

Ties and position flips score 0.5. The `train` row is shown only to expose the Goodhart gap; the `eval` judge is the one that counts.

## Style (judge-independent)

| Metric | Base | Tuned |
|---|---|---|
| mean_words | 285.750 | 265.688 |
| slop_per_1k_words | 1.750 | 2.823 |
| em_dash_per_1k_words | 0.656 | 0.235 |
| list_or_header_share | 0.562 | 0.438 |
| distinct_2 | 0.889 | 0.885 |
| truncated_share | 0.125 | 0.125 |

Mean words delta: -20.1 [-46.1, +3.1]. Stock-phrase delta per 1k words: +1.07 [-0.24, +2.47].

## Preference data

| Stat | Value |
|---|---|
| prompts | 16 |
| candidates | 64 |
| candidates_truncated_share | 0.172 |
| pairs_attempted | 36 |
| pairs_judged | 36 |
| position_consistent_share | 0.528 |
| strict_consistent_pairs | 12 |
| judge_chose_longer_share | 0.333 |
| pairs_after_length_balance | 8 |
| final_chose_longer_share | 0.500 |

## Training

Base `Qwen/Qwen2.5-0.5B-Instruct`, 6 train pairs, 2 held out.

- eval_loss: 0.6397
- eval_rewards/accuracies: 0.5000
- eval_rewards/margins: 0.1535
- eval_rewards/chosen: 0.2389
- eval_rewards/rejected: 0.0853
