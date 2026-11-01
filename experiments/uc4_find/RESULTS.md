# UC4: can a routing adapter help a model find documents faster?

2026-10-31. **All of this is exploratory:** the pre-registered 6-hour limit was already used up, and the UC1 final users had already been seen. The criteria were written into [CRITERIA.md](../CRITERIA.md), with their sha256 recorded, before any baseline ran. All data is public or synthetic. The hardware was one shared A100 (called `a100` below), which is T3-C. This file makes no hardware privacy claim.

## Conclusion

**By the frozen criteria the routing adapter failed: it won on neither seen nor held-out documents, and the outcome is "hypothesis not supported".** By the outcome table written in advance, the next step is that the product goes with catalogs and indexes, with LoRA only for skills.

| Opponent | Pass condition | Seen, 200 questions | Held-out, 60 questions |
|---|---|---|---|
| B2 questions in the index | First-choice hit at least 5 points higher, and CI lower bound > 0 | −29.0 [−36.5, −21.8] **fail** | −61.7 [−73.3, −50.0] **fail** |
| B1 full catalog | First-choice hit no lower than B1 minus 5 points | −30.0 [−37.2, −23.0] **fail** | −58.3 [−70.0, −46.7] **fail** |
| B1 full catalog | Total input tokens per correct answer at least 30% lower | −93.1% [−95.2, −90.6] pass | −59.9% [−83.6, +40.9] pass (point estimate only) |
| B3 one-shot RAG | Accuracy no more than 3 points lower | −0.8 [−8.2, +6.8] pass | −27.5 [−40.8, −14.2] **fail** |
| B3 one-shot RAG | Read tokens per correct answer at least 20% lower | −68.3% [−76.4, −57.6] pass | +117.4% [−4.4, +684.5] **fail** |

| Side-effect line | Figure | Result |
|---|---|---|
| Tool parse rate at least 95% | 100% (1,806/1,806) | Not triggered |
| Refusal no more than 5 points below the best baseline | L1 6.7, B3 100.0, difference −93.3 | **Triggered, cannot ship** |
| Regression at least 72 on every seed | 78, 74 | Not triggered |
| C1 no more than 3 points above B0 | −55.5 [−62.5, −48.5] | Not triggered, the evaluation did not leak |

Three things decide how to read this result:

1. **The adapter did learn, but it learned only the training phrasing.** First-choice hit on the training phrasing was 98.3% and 99.0%. On the frozen test, written by a different generator, it was only 28.5%.
2. **It does not know what it does not know.** On held-out and no-such-document questions, L1's first step guessed a document and went straight to `read` 92.5% and 96.7% of the time, even though the no-such-document phrasing it was trained on went to `search` 97.8% to 100% of the time. This has nothing to do with phrasing style: in E4 below, first-choice hit on held-out documents was 0.
3. **The post hoc exploration E4 points the other way. It is not part of any verdict, but it has to be mentioned.** Tested with Bolt phrasings left out of training, L1 was 7.0 points above B2 [+0.8, +13.5]. The frozen test favors search and E4 favors LoRA. Only the hand-written questions of step 11 can say which side real user questions resemble, and that step is on hold.

Product copy must not say "the model knows your documents".

## Setup

| Item | Detail |
|---|---|
| Corpus | 300 UC3 documents indexed, 240 seen and 60 held-out. Documents of the same domain and type were not split across the two sides. The 40 in `dev.json` were not indexed: 24 produced no-such-document questions for training and 16 for testing |
| `doc_id` | A readable path: domain, a title written by Bolt, last four digits of the file number, for example `legal/contract-dispute-summary-judgment-motion-4455` |
| Training questions | Bolt wrote them, 10 per document. Near-duplicate documents were written with their contrast document alongside |
| Test questions | Qwen3.5-2B wrote them with a different set of prompts, 320 questions: 200 seen, 60 held-out, 60 no such document |
| Tools | `search(query, k=5)` (BM25 plus vector, RRF); `read(doc_id)` (limit 800 tokens, returns an error if the ID does not exist) |
| Loop | transformers, greedy, thinking off, at most 6 turns, batched by turn |
| L1 | rank 16, alpha 32, lr 2e-4, 4 epochs, batch 8, length 256, 2 seeds. The target is the first tool call. Answers and closed-book refusal were not trained |
| Precheck | Retrieval R@1 on the test questions: B0 0.596, below the 0.85 gate, so the questions were not saturated. That gate was used only for this precheck and is not in the frozen criteria |

## All metrics

**Seen documents, 200 questions.** L1 is the two seeds combined.

| Arm | First-choice hit | Accuracy | Turns | Read tokens per question | Total input tokens per question | Read per correct answer | Total input per correct answer | Seconds per question, batched |
|---|---|---|---|---|---|---|---|---|
| B0 bare Bolt | 57.5 | 42.5 | 3.01 | 692 | 1,861 | 1,629 | 4,378 | 0.62 |
| B1 full catalog | 58.5 | 20.0 | 2.65 | 731 | 15,229 | 3,656 | 76,144 | 2.75 |
| B2 questions in the index | 57.5 | 39.0 | 3.01 | 687 | 1,842 | 1,762 | 4,722 | 0.65 |
| B3 one-shot RAG | 58.0 | 29.0 | 1.00 | 1,186 | 1,252 | 4,091 | 4,317 | 0.15 |
| **L1** | **28.5** | **28.3** | 3.28 | 366 | 1,479 | 1,296 | 5,234 | 0.91 |
| C1 placebo | 2.0 | 6.0 | 3.99 | 691 | 2,732 | 11,510 | 45,538 | 1.14 |

By seed: L1 first-choice hit 28.0, 29.0; accuracy 29.0, 27.5.

| Arm | Near-duplicate document hit (77 questions) | First document opened is the target | Invented IDs (share of reads) | Invented IDs (by question) | Recovery rate | Parse rate | Hit the turn limit |
|---|---|---|---|---|---|---|---|
| B0 | 59.7 | 45.5 | 0 | 0 | 16.5 | 99.8 | 2.0% |
| B1 | 59.7 | 26.0 | 33.1 | 7.0 | 6.0 | 100 | 2.0% |
| B2 | 57.1 | 45.5 | 0 | 0 | 14.1 | 100 | 2.0% |
| **L1** | 29.2 | 28.2 | 54.4 | 33.8 | 7.0 | 100 | 25.8% |
| C1 | 2.6 | 1.5 | 24.5 | 36.5 | 6.6 | 100 | 40.0% |

First-choice hit on the search arms only says the top result was the target. The model does not always open it, so the "first document opened is the target" column compares better. L1 also lost to B0 by 17 points in that column.

**Held-out documents, 60 questions.**

| Arm | First-choice hit | Accuracy | Turns | Read per correct answer | Total input per correct answer |
|---|---|---|---|---|---|
| B0 | 65.0 | 36.7 | 2.95 | 1,839 | 5,035 |
| B1 | 61.7 | 18.3 | 2.78 | 3,692 | 87,327 |
| B2 | 65.0 | 35.0 | 2.85 | 1,844 | 4,859 |
| B3 | 65.0 | 33.3 | 1.00 | 3,523 | 3,719 |
| **L1** | **3.3** | **5.8** | 4.14 | 7,659 | 35,038 |
| C1 | 6.7 | 3.3 | 4.30 | 23,083 | 91,874 |

**No such document, 60 questions.** The frozen definition is an answer matching `REFUSAL`. The regex misses phrasings like "The documents do not contain…", so a lenient definition is reported as well. The verdict still uses the frozen definition.

| Arm | Frozen definition | Lenient definition | No answer | Gave a specific answer |
|---|---|---|---|---|
| B0 | 20.0 | 45.0 | 13.3 | 41.7 |
| B1 | 0.0 | 75.0 | 5.0 | 20.0 |
| B2 | 16.7 | 43.3 | 16.7 | 40.0 |
| B3 | 100.0 | 100.0 | 0.0 | 0.0 |
| **L1** | **6.7** | 10.8 | 35.8 | **53.3** |
| C1 | 8.3 | 15.0 | 61.7 | 23.3 |

B3's 100 is not all skill: it also answered UNKNOWN on 67.5% of seen documents. Even compared with B0 alone, L1 was 13.3 points lower (frozen definition), which also fails the 5-point line.

**Regression.** B0 74, L1 78 and 74, C1 77. There was no drop this time. UC2 and UC3 adapters fell to scores of 1 to 5 at the time. This time the replay used the base model's own outputs and included 34 multiple-choice questions, and lr and epochs were also lower. Which of these mattered was not tested separately. **With tools declared**, B0 scored 0 on these 100 questions and L1 scored 5 and 0, because under the system prompt the base model searches or answers UNKNOWN. That is the effect of the prompt, not of the adapter.

**Manipulation check.**

| adapter | First-choice hit on training phrasing | Trained no-such-document phrasing went to search | Training seconds |
|---|---|---|---|
| L1 seed 1 | 98.3 | 100.0 | 769 |
| L1 seed 2 | 99.0 | 97.8 | 768 |
| C1 (shuffled labels) | 95.7 | 94.3 | 770 |

All passed the 90% gate, so nothing was retrained. This failure was not "the recipe did not learn".

**Serial latency, 40 questions, seconds per question (shared GPU, PEFT not optimized, compare relative values only).**

| Arm | Mean | Median | Turns | Read the target |
|---|---|---|---|---|
| B0 | 3.38 | 3.37 | 2.75 | 47.5% |
| B1 | 4.99 | 4.77 | 2.67 | 37.5% |
| B2 | 3.42 | 3.48 | 2.88 | 55.0% |
| B3 | 0.39 | 0.19 | 1.00 | 82.5% (target in the top 5) |
| L1 | 7.37 | 4.38 | 3.38 | 36.2% |

L1 was not faster. It was slower than B0: more turns, and each turn was slower too.

## Why it failed

**Where L1's first call went (seen documents, 400 calls).**

| Where it went | Share |
|---|---|
| Hit | 27.0% |
| Went to search | 4.0% |
| Read an existing document, right domain, wrong document | 21.3% |
| Read an existing document, wrong domain too | 24.5% |
| ID does not exist | 18.3% |
| Title and number right, domain prefix wrong, ID does not exist | 5.0% |

**Failure cases, verbatim.** All are from synthetic data.

| Question | Target | L1's first call |
|---|---|---|
| Which team is looking after the car with plate PQM 738? | `automotive/automotive-trade-in-form-details-8755` | `read` another document in the same domain, `automotive/vehicle-warranty-service-report-4503` |
| How many days is the review period for the film about the art of filmmaking? | `media/cinematography-video-review-insights-3662` | `read` `entertainment/cinematography-video-review-insights-3662`, wrong prefix, ID does not exist |
| Who is handling the credit card processing and network address issues? | An indexed document | `read` 6 e-commerce documents in a row and hit the turn limit |

**After a wrong guess it did not come back.** Training covered only the first call. The step after reading a document was never trained. After a wrong guess L1 kept guessing: 25.8% of questions hit the 6-turn limit, and the recovery rate was 7.0% against 16.5% for B0. L1s showed this: with the adapter on the first call only, the recovery rate came back to 23.4%.

**The wrong document also looks right.** Every document has a review window and a team in the same sentence pattern, so a wrong document can still produce a confident answer. This is a feature of this corpus. Real documents may not be so uniform.

## What the baselines showed

- **B1 did not really use the catalog.** Under the original prompt all 320 questions got a direct UNKNOWN, and that run was voided. Under the selected P2 prompt the model still used `search` as its first step 100% of the time, so the 5,108-token catalog was paid for every turn for nothing, and reading was worse too: accuracy 20.0, with more than half of the hits never opening the document. So "L1 lost to B1" really means it lost to search. **The hit rate of the catalog itself was not measured this time.**
- **B2 did not help the agent.** Writing questions into the index raised R@1 on the raw questions from 0.58 to 0.64, but once the model rewrote its own queries, first-choice hit was the same as B0 (4 questions better, 4 worse) and accuracy was 3.5 points lower. R@1 on raw questions for held-out documents fell from 0.65 to 0.517: documents with added questions pushed down those without. This method would hurt new documents.
- **B3 lost on reading.** The target was in the top 5 for 78% of questions, it answered UNKNOWN on 67.5% of seen documents, and accuracy was only 29.0. This matches the UC3 diagnosis: retrieval is not the bottleneck, reading is.
- **B0 was already cheap.** 3 turns on average, 4,378 tokens per correct answer.

**The test questions have a defect. It was found only after seeing outputs, so by the rules it was not changed.** Qwen's fixed opening "Who is handling" makes up 29% of the questions on indexed documents (76/260). The model naturally answers with a person named in the document, while the gold answer is the internal team. When B0 read the target document, accuracy was 54.9% for this opening and 92.6% to 100% for the others. All arms are affected the same way, so the comparison still holds, but absolute accuracy is on the low side.

## Post hoc exploration (not part of any verdict)

### L1s: adapter on the first call only

On the frozen test, first-choice hit is the same as L1 (the same call by the same adapters). After that the base model takes over.

| Part | Metric | B0 | L1 | L1s |
|---|---|---|---|---|
| Seen | Accuracy | 42.5 | 28.3 | 41.7 |
| Seen | Turns | 3.01 | 3.28 | 2.70 |
| Seen | Total input tokens per correct answer | 4,378 | 5,234 | 3,069 |
| Held-out | Accuracy | 36.7 | 5.8 | 18.3 |
| No such document | Refusal (frozen definition) | 20.0 | 6.7 | 40.0 |
| No such document | Refusal (lenient definition) | 45.0 | 10.8 | 45.8 |
| All | Parse rate | 99.8 | 100 | 99.5 |

L1s − B0, seen documents: accuracy −0.8 [−7.5, +6.0], total input per correct answer −29.9% [−42.4, −15.0]. Held-out documents: accuracy −18.3 [−30.0, −7.5]. One way to read this: a right guess saves a turn of search, a wrong guess can still be rescued by the base model, and the price is worse results on new documents. Doing this online would need the gateway to switch adapters per call, and that was not verified.

### E4: Bolt phrasings left out of training

This was decided only after seeing the gap between 98% and 28%. One question was drawn per document with the same filter. These questions come from the same generator as the training questions, so this is not a clean test. It only serves to separate "fails when the phrasing changes" from "fails only when the generator changes".

| Part | Metric | B0 | B2 | B3 | L1 | L1s |
|---|---|---|---|---|---|---|
| Seen, 200 questions | First-choice hit | 49.0 | 74.5 | 50.5 | 81.5 | 81.5 |
| Seen | Accuracy | 44.5 | 60.5 | 26.0 | 62.3 | 75.0 |
| Seen | Turns | 3.15 | 3.08 | 1.00 | 2.61 | 2.18 |
| Seen | Total input tokens per correct answer | 4,618 | 3,192 | 4,827 | 1,713 | 1,069 |
| Held-out, 60 questions | First-choice hit | 41.7 | 36.7 | 38.3 | **0.0** | **0.0** |
| Held-out | Accuracy | 48.3 | 50.0 | 16.7 | 4.2 | 31.7 |
| No such document, 60 questions | Refusal (lenient definition) | 38.3 | 36.7 | 100.0 | 13.3 | 39.2 |

| Comparison, seen documents | First-choice hit | Accuracy | Total input tokens per correct answer |
|---|---|---|---|
| L1 − B2 | +7.0 [+0.8, +13.5] | +1.8 [−5.5, +9.0] | −46.3% [−54.9, −36.2] |
| L1s − B2 | +7.0 [+0.8, +13.2] | +14.5 [+7.5, +21.5] | −66.5% [−71.6, −60.8] |
| L1 − B0 | +32.5 [+24.8, +40.2] | +17.8 [+9.8, +25.8] | −62.9% [−70.2, −54.6] |

Looking only at B0 suggests a 32-point win. Against B2, the opponent the criteria name, the win was only 7 points. B2 also used the same questions. This comparison was added before the conclusion was written.

**The two question sets differ in wording.**

| Questions | Words found in the target document | Words found in this document's training questions | B0 | L1 |
|---|---|---|---|---|
| Frozen test (Qwen) | 68.8% | 43.0% | 57.5 | 28.5 |
| E4 (Bolt) | 25.1% | 62.1% | 49.0 | 81.5 |

| Three bands by wording | B0 | L1 |
|---|---|---|
| Frozen test, overlap with training questions 0.00 to 0.25 | 37.9 | 7.6 |
| Frozen test, overlap with training questions 0.67 to 1.00 | 70.1 | 48.5 |
| E4, overlap with the document 0.00 to 0.14 | 9.1 | 76.5 |
| E4, overlap with the document 0.33 to 1.00 | 85.1 | 88.1 |

Qwen's questions copy document details, which is home ground for search. Bolt's questions avoid the document's wording and stay close to the training questions, which is home ground for LoRA. When a question contains almost none of the document's own words, search nearly fails and L1 still reaches 76.5%. This is the closest signal in all the experiments to "the model knows which documents you have". But the signal depends on the phrasing resembling the training questions, and the mechanism may be nothing more than memorizing similar sentences.

**What E4 did not rescue:** first-choice hit on held-out documents was 0, and on no-such-document questions the first step went to `search` only 5.8% of the time. Whatever the phrasing, the adapter could not tell whether it knew the document. The old closed-book refusal problem came back in the form of closed-book routing.

## Expectations stated in advance, right and wrong

| Expected | Actual |
|---|---|
| L1 has room on seen documents | Wrong, 29 points lower |
| Failing to generalize after the generator changes is the main risk | Right, 98% to 28.5% |
| The answer after reading a document is at risk | Right, 25.8% hit the turn limit |
| Regression is at risk | Wrong, 78 and 74, no drop |
| L1 does not win on held-out documents | Right |
| Saving tokens against B1 is not a problem | Right, but B1 did not really use the catalog |

## Differences from the task brief

- The criteria were frozen before the baselines, earlier than step 8. Later changes were append-only, each with a node hash. The body of the criteria and the thresholds in this experiment's section did not change. The hash at freezing was `5211d5c8…`, and the hash of the first 9,753 bytes of the criteria body is still `7d0fb635…`. Both hashes are of the original text of the criteria, not of the translation in this repository.
- The system prompt plus tool declarations were about 250 tokens at first and did not fit the 256 limit. They were cut to 169 and the smoke test was rerun. Each pair was about 230 tokens, not the estimated 190.
- There were 2,967 pairs (`read` 2,377, `search` 230, replay 360), not 3,000. The shortfall came from documents with fewer than 10 questions after filtering.
- Without prefill, Qwen 2B wrote questions as answers, and with examples added it copied the examples, so fixed question openings were used instead.
- The filter was tightened before any arm answered questions: number-shaped identifiers and canary names. A Bolt validator was added as well, keeping only questions whose described content really appears in the document. 6 no-such-document questions did not go through the validator.
- **The manual spot check was not done by a person.** I (Claude) read 30 questions together with their documents: 29 had the described content in the document, 1 contained a canary name and the filter was fixed, and 2 were rather generic. No new sample was drawn after the fix.
- After B1's original prompt failed, P2 was chosen using training questions only. The rule was registered first, and only B1 was rerun.
- Peak GPU memory for a single training process was 40.3GB, not the estimated 20GB. The process limit was changed to 44GB; logits alone took 1.7GB at a time. During training the GPU had about 17GB left, below the 24GB that Bolt training in the matrix waits for, but a single training run took only about 13 minutes, the matrix can wait up to 1 hour, and the M1, R1 and R2 logs also ended before this. No one else's process was killed.
- The analysis script at first excluded questions with no tool call in the first turn from first-choice hit, which did not match the criteria. This was fixed before the results were written. Key numbers from an independent recalculation agreed.
- In the smoke test the R3 adapter did not call a tool once in 20 questions. This was the first measurement showing that a closed-book adapter does not use tools once they are declared. "Parse rate ≥ 90%" has no denominator for it, so it was recorded as not passed, with the cause known.

## Not tested

- **Real user questions.** Step 11 is on hold, see below.
- **A B1 that really picks documents from the catalog.** The model always went to `search`. A catalog with a one-line topic was not tested either; it would probably exceed the 8,192 context limit.
- **Decay after files change, the online vLLM path, and thinking turned on.**
- **Recipe changes.** rank, lr, epochs and questions per document were not swept. Training phrasing already reached 98%, so what is missing is diversity, not fit.
- **Routing with a fallback.** For example, read directly only when confidence is high and search otherwise. L1s is only a rough version.
- **A clean version of E4.** It was done post hoc, with the same generator as the training questions.
- There were only 2 seeds and 1 corpus, so the CIs reflect question sampling only. The corpus is uniform: every document has facts in the same sentence patterns.

## Step 11: on hold, not run

`L1_real` was trained before any hand-written question existed (62 documents, 790 pairs, first-choice hit on training phrasing 86.0%, 241 seconds). A trial run of the pipeline on training questions passed. The trial results were deleted and are not results for real questions. The corpus is 62 documents, not the estimated 66 or so, because there were duplicate files and only 2 documents in `code/beetle` are mainly in English.

The local files are all in gitignored directories:

- `data/real/catalog.txt`: `doc_id` and title of the 62 documents
- `data/real/questions_template.csv`: an empty table of 30 rows, columns `question,doc_id`

Whoever writes the questions should do this: first write, from memory, questions you would really ask. Then find in the catalog which document holds the answer and fill in `doc_id`. Each question maps to one document only. Do not copy file names or titles. Include both specific and vague questions. Save as `data/real/questions.csv`, then:

```bash
rsync -az experiments/uc4_find/data/real/questions.csv <gpu-node>:beetle-exp/code/uc4_find/data/real/
ssh <gpu-node> 'cd ~/beetle-exp/code/uc4_find && ../bg.sh uc4real python real_check.py eval'
```

About 3 minutes. It reports first-choice hit, the first document opened, turns and input tokens for B0, B1, L1 and L1s. With 30 questions the CI is about plus or minus 18 points, so read the direction only. **How to use the result:** if L1 or L1s is clearly above B0, real questions look more like E4 and routing with a fallback is worth pursuing. If they are clearly below B0, real questions look more like the frozen test and this line of work ends here.

## Cost

The GPU is priced as exclusive use, $1.98/hour. Everything went into the ledger with stage prefix `uc4/`.

| Item | Minutes |
|---|---|
| Data generation (titles, questions, validation, replay, targets) | 77.4 |
| Training 4 adapters (including 2 GPU memory failures, estimated at 5 minutes) | 47.5 |
| Manipulation check and regression | 6.8 |
| Evaluation (including serial latency, B1 prompt selection, E4) | 113.1 |
| Step 11 question generation and trial run | 12.8 |
| Other (loading, smoke test, precheck) | 3.3 |

| Basis | Hours | USD |
|---|---|---|
| Actual occupancy (union of time intervals) | 3.49 | $6.92 |
| Sum of items | 4.35 | $8.61 |

This is within the 6-hour limit. A single adapter took about 769 seconds, $0.42, 2.43M tokens and 1,484 steps, and is 130MB in size. All 4 adapters are saved in `adapters/uc4/` in the node's working directory. The node stayed on throughout the experiment, and that time is not counted as GPU occupancy.

## Reproducing

```bash
HOST=<gpu-node> experiments/sync.sh
# In ~/beetle-exp/code/uc4_find on the node, run detached from ssh with ../bg.sh NAME CMD:
python -m pytest -q test_harness.py
python run.py smoke
python gen_data.py ids && python gen_data.py train_q && python gen_data.py test_q && python gen_data.py test_q_low
python gen_data.py verify && python gen_data.py tests        # then train_q_short, then verify and tests, repeated once
python gen_data.py replay && python gen_data.py targets && python gen_data.py bolt_tests
python run.py precheck
python run.py eval --arms B0,B2,B3 --tag final && python run.py tune_b1 && python run.py eval --arms B1 --b1-prompt P2 --tag final
python run.py train --name L1_s1 --seed 1 && python run.py train --name L1_s2 --seed 2 && python run.py train --name C1 --seed 1 --shuffle
python run.py eval --arms L1_s1 --adapter L1_s1 --tag final     # same for L1_s2 and C1; for L1s add --first-turn-only
python analyze.py final && python analyze.py bolt && python analyze.py overlap
```

This needs the UC3 data, the cached document vectors and a local copy of Nemotron-PII. Everything uses transformers and does not depend on vLLM. The generated questions depend on model output and batch composition, so their content changes on a rerun. The construction method and the grading do not. Data and per-question logs are not published with the repository.
