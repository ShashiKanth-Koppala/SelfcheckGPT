# Self-Consistency Hallucination Detection (SelfCheckGPT-style)

A small experiment testing whether an LLM's self-consistency across multiple generations predicts whether it's hallucinating using biographies of 20 real but lesser-known notable people, verified against Wikipedia.

## Hypothesis

If a model actually "knows" a fact, it should express that fact consistently across independent samples. If it's hallucinating, different samples should drift, contradict each other, or hedge differently because there's no real underlying knowledge anchoring the generation, just plausible-sounding text.

This is the core idea behind SelfCheckGPT (Manakul et al.): you don't need access to model internals or a reference answer to flag likely hallucinations — you just need to sample the same prompt multiple times and measure how much the outputs agree with each other.

The obvious limitation, which this project ended up demonstrating directly rather than just assuming: a model that is *consistently* wrong will pass a self-consistency check every time, since agreement across samples isn't the same thing as agreement with reality.

## Setup

- 20 real, Wikipedia-documented but not household-famous people (scientists, Nobel laureates, one politician)
- 5 biography samples generated per person, locally, via Ollama (llama3), at temperature 0.8
- Ground truth: scraped Wikipedia intro/article text via the Wikipedia API

## Pipeline

![Pipeline architecture](selfcheckgpt_pipeline_architecture.png)

1. **Scrape ground truth** — pull each person's Wikipedia article text via the Wikipedia API, split into paragraphs, filter out section headers and short fragments.
2. **Generate biographies** — same prompt, 5 samples per person, non-zero temperature so samples can actually diverge.
3. **Extract atomic facts** — decompose each biography into single, checkable claims (one date, place, or achievement per fact) using an LLM extraction pass.
4. **Score consensus** — embed all facts per person (sentence-transformers), cluster near-duplicate phrasings, and compute what fraction of the 5 samples support each fact that was mentioned more than once. Facts mentioned only once are excluded from scoring — they carry no consistency information either way, and including them was found to conflate "hallucination" with "one sample being more verbose than the others."
5. **Detect contradictions** — an early version tried to flag contradictions using embedding similarity bands (facts that are topically close but phrased differently). This didn't work well: cosine similarity captures topical relatedness, not logical conflict, so it mostly caught paraphrases of the same fact and missed real conflicts (e.g. two different named institutions for the same degree). Replaced with an NLI model (`cross-encoder/nli-deberta-v3-base`) that directly classifies whether one fact contradicts another.
6. **Verify against ground truth** — for each fact, find the most relevant Wikipedia paragraph(s) via keyword overlap, then ask a local LLM to judge the fact as TRUE, FALSE, or UNKNOWN based only on that paragraph.
7. **Correlate** — compare each person's consensus/contradiction scores against their actual FALSE-rate from step 6.

## Result

| Person | FALSE-rate | Consensus | Contradiction |
|---|---|---|---|
| Chien-Jen Chen | 0.867 | 0.486 | 0.143 |
| Luis Walter Alvarez | 0.447 | 0.640 | 0.000 |
| Ada Yonath | 0.278 | 0.960 | 0.000 |
| Chien-Shiung Wu | 0.275 | 0.667 | 0.000 |
| Nikolai Vavilov | 0.241 | 0.550 | 0.000 |
| Grace Hopper | 0.204 | 0.683 | 0.000 |
| Abdus Salam | 0.188 | 0.560 | 0.000 |
| Har Gobind Khorana | 0.174 | 0.660 | 0.200 |
| Wangari Maathai | 0.171 | 0.711 | 0.000 |
| Lise Meitner | 0.170 | 0.600 | 0.000 |
| Rachel Carson | 0.152 | 0.636 | 0.091 |
| Katherine Johnson | 0.140 | 0.533 | 0.067 |
| Dorothy Hodgkin | 0.128 | 0.833 | 0.000 |
| Vera Rubin | 0.122 | 0.489 | 0.000 |
| Rosalind Franklin | 0.111 | 0.617 | 0.083 |
| Barbara McClintock | 0.100 | 0.740 | 0.000 |
| Norman Borlaug | 0.100 | 0.637 | 0.000 |
| Subrahmanyan Chandrasekhar | 0.073 | 0.745 | 0.000 |
| Emmy Noether | 0.073 | 0.567 | 0.000 |
| Srinivasa Ramanujan | 0.042 | 0.717 | 0.000 |

Pearson correlation across all 20 people:
- FALSE-rate vs. consensus rate: r = -0.252 (expected direction, not significant at n=20)
- FALSE-rate vs. contradiction rate: r = 0.367 (expected direction, not significant at n=20)

## What actually worked

Chien-Jen Chen had both the lowest consensus and the highest contradiction rate in the dataset — and also the highest FALSE-rate by a wide margin (0.867). His biography samples confused him with fabricated names and mismatched roles across generations. This is the case the method is designed to catch, and it caught it clearly.

## What the method missed

Two people — Luis Alvarez and Ada Yonath — had high FALSE-rates (0.447 and 0.278) despite showing little to no contradiction and, in Yonath's case, the *highest* consensus score in the entire dataset (0.96). In both cases the model was repeating the same wrong facts confidently across all 5 samples. Self-consistency has no way to catch this, since the model agrees with itself every time — it's just agreeing on something false.

This is the known blind spot of self-consistency-based hallucination detection: it detects uncertainty, not truth. A model that is confidently and consistently wrong is invisible to it. Seeing this show up twice in a sample of 20 is a useful, concrete confirmation of a limitation that's usually only discussed theoretically.

## Other things worth knowing if reproducing this

- The Wikipedia verifier rarely outputs UNKNOWN, even for facts not covered by the matched paragraph. This suggests the model is sometimes falling back on its own training knowledge rather than strictly using the provided reference text, despite being instructed not to. Worth tightening the prompt or trying a different verifier model if reproducing this.
- Fact extraction and verification both run through the same class of local model that generated the original biographies. This isn't a fully independent check — a systematic blind spot in the model could show up at every stage. Using a different, larger, or closed-source model for verification would be a more rigorous setup.
- The NLI-based contradiction check is a real improvement over the embedding-similarity version, but it isn't perfect either — short, decontextualized fact fragments (e.g. "Biophysicist" vs. "Born in London") sometimes get flagged as contradicting when they're just unrelated. NLI models are trained on full sentences, not clipped fact fragments, so some noise here is expected.

## Possible next steps

- Compare against RAGAS-style faithfulness scoring on the same eval set
- Use a stronger/independent model for the verification step
- Try reconstructing full sentences before the NLI contradiction check, to see if it reduces the fragment-related false positives

## AI USAGE
I have used LLM's to assisnt myself in generating the code. All the results have been manually verified before commiting. If any discrepancy is found please raise an issue and would right away rectify it.


## References

- [SelfCheckGPT (Manakul et al., 2023)](https://arxiv.org/abs/2303.08896) — the paper this project is based on
- [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) — the NLI model used for contradiction detection
