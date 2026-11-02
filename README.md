# BenchBrief

**Decision support for judges: a case summary, cited precedent, and the reasoning paths that precedent supports. Never a verdict.**

BenchBrief is a retrieval-augmented generation (RAG) system over federal opinions from the U.S. District Court for the Northern District of California (sourced from [CourtListener](https://www.courtlistener.com/)). A judge pastes a case or asks a follow-up question; the system retrieves relevant passages from the corpus, answers **only** from those passages, and cites each claim. It is a research aid, **not legal advice**.

> **Status:** working end to end on a deliberately small corpus (12 opinions, 243 chunks). The focus of the project is the engineering around retrieval, grounding and *measurement*, not corpus size.

---

