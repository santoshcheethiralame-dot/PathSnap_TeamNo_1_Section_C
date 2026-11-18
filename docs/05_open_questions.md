# 05 · Open questions

Ask the organisers. Until answered, use the working assumption.

| # | Question | Working assumption |
|---|---|---|
| 1 | Deadline and the exact event schedule? | Unknown. Plan for a demo-able end-to-end baseline early. |
| 2 | What does the **first-phase evaluation (before lunch)** check? | Possibly the API/UI demo. Have the baseline pipeline running first. |
| 3 | How do they derive **per-point reference labels** from `.route` when the API scores accuracy? | Ours come from a revisit-aware monotone alignment (`03_plan.md` §4). If theirs differs, our Top-1 will differ. |
| 4 | Exact definition of **macro F1**? | Binary on candidate rows (is-reference vs not), averaged over the two classes. Class-per-`arc_id` is meaningless because ids are per trip. |
| 5 | Exact definition of **route continuity**? | Fraction of consecutive distinct predicted arcs where the head node of one is the tail node of the next. Report it for the reference route too (it is not always 100%). |
| 6 | Are the ten evaluator trips handed over on the day, in the same four-file format, and where do they go? | Same format, dropped in a server-side folder the API reads. |
| 7 | How should the reference-route defects in training trips 33 and 54 be treated? | Quarantine from training and document (`02_data_audit.md`). |
| 8 | What is the AI-use disclosure format "as required by the event's policies"? | Disclose in README and report. |
