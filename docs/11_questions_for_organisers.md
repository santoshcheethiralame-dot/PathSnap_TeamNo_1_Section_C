# 11 · Questions for the organisers

Ready to send. Priority 1 changes what we build or how numbers are scored; priority 2 is logistics. For each we state our working assumption so a one-line "yes, as assumed" is enough. Context in `05_open_questions.md`.

## Priority 1: scoring and rules
1. **Per-point reference labels.** The API scores accuracy against the reference route (`.route`, a list of arcs). How is the arc that each GPS point "belongs to" derived? We use a monotone alignment that allows revisits. *Assumption: ours.* If yours differs, our Top-1 will differ.
2. **Macro F1.** What exactly is it averaged over? *Assumption: binary per candidate row (is this the reference arc, yes or no), averaged over the two classes.* Arc ids are per trip, so a class per arc id would be meaningless.
3. **Route continuity and sequence-level performance.** How are they defined? *Assumption: continuity = share of consecutive distinct predicted arcs where one ends at the node where the next starts; we also report length-weighted route overlap with the reference route.*
4. **Defective references.** Training trips 33 (route covers only part of the drive) and 54 (many fixes far from every supplied road) look defective. May evaluator trips have the same problem, and are they scored against the reference as given? *Assumption: yes, scored as given; we quarantine 33 and 54 from training and document it.*
5. **AI-assistance disclosure.** What format and where (README, report, both)? Is AI help with code and drafting allowed when disclosed, and does the report text need to be the students' own wording? *Assumption: allowed with disclosure; we rewrite report text ourselves.*
6. **Use of student-test.** We use the 10 student-test trips for reporting and for the development-stage model comparison, and for decoder (HMM) parameter tuning, as the rules allow, but never for fitting or Bayesian search. Is that the intended use?

## Priority 1: the live demo
7. **Evaluator trips on the day.** Same four files per trip (`.track`, `.nodes`, `.arcs`, `.route`)? Where do they go, and is `.route` present on the API laptop (our scoring reads it server-side only)? *Assumption: same format, dropped in a folder on Laptop 1; we point the API at it with an environment variable.*
8. **First-phase evaluation (before lunch).** What does it check (API, repo, report, a model run)? *Assumption: a runnable end-to-end pipeline and the repo.*
9. **Network for the two laptops.** Will both be on the same network with no client isolation, and which ports are allowed? Will there be internet on the UI laptop? Our UI loads the Leaflet library and map tiles from the internet; if there is none the map does not render at all (we can bundle both locally if needed). *Assumption: same Wi-Fi, internet available.*
10. **Per-trip time limit.** Is there a time limit per prediction in the demo? Ours takes 0.3 to 2.4 s per student-test trip and about 7 s for the largest training trip (19,313 points).

## Priority 2: repository and report
11. **Repository contents.** Is a small saved model (1.4 MB `joblib` file) in the repo fine, and are derived artefacts such as one Folium HTML map of a student-test trip allowed given that the dataset must not be published?
12. **Report format.** Page limit, template, and how many Folium screenshots and UI screenshots are expected. Who uploads it and by when?
13. **Deadline and schedule.** Exact deadline for repo and PDF, and the order of the event day.
