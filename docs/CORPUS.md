# Demo Corpus: Apollo 11 Moon Landing

The corpus lives in [`data/`](../data/). This description is kept here, outside `data/`, because every file under the data directory is indexed as retrievable content.

This is a small, original, factual corpus written for this project's Hybrid RAG demo. It was composed from general knowledge specifically for this repository — it is not scraped, copied, or adapted from Wikipedia or any other single source. It replaces the earlier fictional demo document ("The Silence at Ridge 42") with real-world content so retrieval and citation behavior can be evaluated against verifiable facts, including questions that require combining information from two different files (multi-hop questions).

## Files

- `01-mission-overview.md` — What Apollo 11 was: its goal, launch date, administering agency (NASA), and its place in the broader Apollo program.
- `02-crew.md` — The three-person crew (Neil Armstrong, Buzz Aldrin, Michael Collins): their roles, backgrounds, and who walked on the Moon versus who stayed in lunar orbit.
- `03-spacecraft.md` — The hardware: the Saturn V rocket, the Command and Service Module Columbia, and the Lunar Module Eagle.
- `04-timeline.md` — The mission chronology: launch, lunar orbit insertion, undocking, landing, moonwalk, ascent, and splashdown, with approximate UTC times and dates.
- `05-landing-and-moonwalk.md` — The landing itself, the "one small step" quote and its context, and the activities carried out on the lunar surface.
- `06-legacy.md` — The mission's scientific results, its place in the rest of the Apollo program, and its lasting cultural and technical influence.

Each file cross-references the others by filename (e.g., `04-timeline.md` refers back to `02-crew.md` and `03-spacecraft.md`) so that questions spanning two documents have a genuine answer to be retrieved and synthesized, rather than being answerable from a single chunk.
