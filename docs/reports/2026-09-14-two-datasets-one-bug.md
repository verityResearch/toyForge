# Two datasets, one bug

Every row toyForge's verifier had ever graded, before this week, came from the same
place: a hand-written seed, or a teacher-model expansion of one, both authored by
people who type in English and don't think much about it. That's not a criticism —
it's just what "test data" tends to mean when nobody goes looking for trouble. This
is what happened when we went looking.

## The clean run

The first pass was almost aggressively boring. A template teacher — no API calls,
no sampling, just deterministic string substitution — took the 31 hand-seeded
trajectories and expanded them 40 ways each: swap the requester name, swap the
agent, bump the ticket number, append a case tag to the situation text. 1,240 rows,
twelve worker threads, done in well under a second.

Zero rejects. Every row passed the verifier — parse, schema, method, precondition,
transition, all green. Feed those same 1,475 steps through all three reward
presets (`shaped`, `binary`, `schema_only`) and you get a flat line: reward 1.000,
every step, every preset. Nothing interesting happens, and that's *correct* —
this is data that was already accepted by the same verifier during generation, so
of course it scores perfectly under the reward function that shares its logic.
It's a consistency check, and it passed. Good. Boring. Necessary.

But a flat histogram from synthetic data doesn't tell you much about what the
verifier does with input it didn't design for itself. The seeds all say things
like `"A customer reports that the office printer is offline."` Clean ASCII,
predictable sentence shape, `body_text` values that are always comfortably longer
than any schema minimum. If there's a bug lurking in how the C port measures
string length, sentences engineered by the same mind that wrote the schema are
the last place it'll show up.

## Going outside

So: what does the pipeline do with text nobody wrote for it? A handful of live
fetches — Wikipedia's random-article endpoint, Hacker News' front page, an arXiv
listing —
pulled back a small, genuinely uncurated grab-bag: a stub about a defunct Polish
railway station (*"Rozłazino is a non-operational PKP railway station in
Rozłazino (Pomeranian Voivodeship), Poland."*), a paragraph on a rare Tajik
mineral, real HN titles with their colons and parentheses, a paper
title punctuated with a question mark. None of it was picked for any property
except being real — whatever came back from `Special:Random` and the front page
at that moment. (One honest caveat: the fetch tool passes pages through a small
model before returning them, so "verbatim" means as verbatim as that allows.)

A second teacher swapped this scraped text into the same 31 seeds, same expansion
count, same everything else — isolating one variable. 1,240 rows again. Zero
rejects again.

That second zero looked, at first glance, like the boring result repeating
itself. It wasn't. The schema's `body_text` field has exactly one content
constraint — `minLength: 4` — and every scraped snippet cleared it by a wide
margin, sentences and headlines being what they are. A flat pass rate here didn't
mean nothing was tested; it meant the one probe positioned to catch something
never got close enough to the edge to fire.

## Building the edge on purpose

That's the tell that a random sample, however genuine, isn't the same tool as a
constructed adversarial case. If the schema's only content check is a length
threshold, then the test that actually matters is a string sitting *exactly* on
that threshold — and JSON Schema's `minLength` is specified in Unicode code
points, not bytes. Most of the time nobody notices the difference, because most
text is ASCII and a code point is a byte. The string `łł`, two Polish letters, isn't: two code
points, four bytes, because `ł` takes two bytes in UTF-8.

Feed `body_text: "łł"` against `minLength: 4` and Python's `jsonschema` — the
project's own oracle, the thing every C output gets checked against — correctly
says: too short, 2 code points, reject. Feed the identical string to the C
verifier — the one that actually runs during evaluation and reward-scoring, not
the reference implementation — and it said: fine, schema valid, `1.0`.

That's a real, confirmed disagreement between the two implementations this
project is built to keep byte-for-byte identical. `c/src/jsonschema.c` turned out
to hold two separate length-measuring code paths. The general Draft 2020-12
engine — the one exercised by the vendored conformance suite — already does this
correctly, with a helper that walks the decoded bytes and counts everything
that isn't a UTF-8 continuation byte. The other path, `validate_string_constraints`
— the fast, hand-rolled validator that the actual training and evaluation hot
path calls, not the general-purpose one — used `strlen()`. Byte count. Nobody
had gone back and connected the two.

The fix is one call site and one small helper: a code-point counter over an
already-decoded C string, reusing the exact technique the correct
implementation already used, just adapted to a plain string instead of a raw
JSON span. Rebuilt, re-run: `łł` correctly fails now, `łłłł` (four code points,
eight bytes — the boundary case just past it) correctly passes, and nothing else
moved. All 25 CTests, all 17 C-versus-Python differential parity gates, the
ASan/UBSan build, and the full pytest suite came back clean afterward. None of
those suites had ever put a multi-byte string right at a length boundary, so none
of them had ever had a reason to notice the two implementations disagreed.

## What each dataset was actually for

It's tempting, after finding a real bug, to conclude the synthetic run was a
waste of time and the scraped one was the "real" test. That's not quite right,
and the flat-1.000 histogram is worth defending rather than dismissing. The
synthetic dataset answered a question the scraped one structurally couldn't:
does the reward function agree with the verifier that already gated this data
into existence? If those two diverge on data the pipeline itself produced, the
bug is in the relationship between eval-time grading and RL-time reward — a
different and arguably scarier failure mode than a Unicode edge case, because it
would mean the thing training the model doesn't measure the same thing as the
thing certifying the model. Ruling that out cheaply, at volume, before looking
anywhere else, is real work even when the answer is "yes, they agree."

What the scraped run bought wasn't volume or realism for its own sake — eleven
snippets of Wikipedia, Hacker News and arXiv don't constitute a stress test by weight.
What it bought was *not having chosen the content*. Every synthetic test case in
this project, including the deliberately adversarial `WRONG_METHOD` and
`MALFORMED` cases from the reward-shaping comparison earlier the same session, was written
by someone who already knew what they were trying to prove. That's not a flaw —
it's what a unit test is — but it means the space of "things I remembered to
check" and the space of "things a stranger's sentence happens to contain" are
different spaces, and only one of them includes a Polish railway station's name
landing exactly where a length boundary could be probed. The scraped run didn't
find the bug by being adversarial. It found the ingredient — a real multi-byte
character in a real field — that made constructing the adversarial case possible
in the first place. The random sample was the tip-off; the four-line reproduction
that actually pinned the bug down still had to be built on purpose, once there
was a concrete reason to build it.

Neither run alone would have gotten here. The clean set proved the reward
pipeline is internally consistent. The scraped set proved the verifier isn't
internally consistent with *itself* — two implementations of the same
specification, quietly disagreeing, for as long as nothing non-ASCII ever
reached the hot path. Both of those are load-bearing facts about a training
signal a model is going to be shaped by. Only one of them was sitting there to
be tripped over by accident.
