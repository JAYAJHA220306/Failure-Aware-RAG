# Corpus

Put the documents the system may answer from here (.pdf, .md, .txt). Files named README are skipped.

The five demo files are simplified test texts, **not clinical guidance**. `forum_post_unverified.md`
is deliberately wrong so the conflict repair can be demonstrated. Replace them with real guidelines
(for example NICE, WHO or ICMR guideline PDFs) before running experiments.

Optional front-matter at the top of a Markdown/text file:

```
---
title: Hypertension in adults (NICE NG136)
authority: guideline   # guideline | textbook | article | forum
---
```

PDFs cannot carry front-matter, so their authority comes from the file name: a name containing
`guideline`, `textbook` or `forum` gets that authority (e.g. `nice_hypertension_guideline.pdf`);
anything else is `article`.
