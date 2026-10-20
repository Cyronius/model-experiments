# Title format

Canonical spec for how titles are cleaned and validated, for both teacher-generated training
labels and the shipped model's output. Code: `scripts/common/validation.py`,
`scripts/common/lmstudio_client.py`. Tests: `specs/title/tests/` (run with `uv run pytest`).

### TITLE-MAX-WORDS: Over-long teacher titles are retried, never cut off
**Applies to:** titler
**Verification:** test

When titling training data, a teacher reply longer than 8 words, or one stopped by the token
limit, is rejected and retried. It is never shortened, because cutting leaves broken endings
("Why Some People Don't Want Their Dogs To").

**Acceptance criteria:**
- Teacher replies with 11 words, then a 4-word title → the 4-word title is kept, on attempt 2
- Teacher reply ends because of the token limit, then a valid title → the valid title is kept, on attempt 2
- `validate_title` on an 11-word title → `"too long (11 words)"`

### TITLE-PRODUCT-CLAMP: Model output is clamped to 8 words
**Applies to:** titler
**Verification:** test

Titles shown to users (model output in eval, examples and the web demo) are cut to the first 8
words by `clean_title`.

**Acceptance criteria:**
- `clean_title` on an 11-word title → its first 8 words

### TITLE-APOSTROPHE: Apostrophes inside words are kept
**Applies to:** titler
**Verification:** test

`clean_title` keeps apostrophes inside words ("Don't", "Children's") and strips every other
quote mark. A curly apostrophe becomes a straight one. `validate_title` accepts in-word
apostrophes and rejects other quote marks.

**Acceptance criteria:**
- `“Fixing My Children’s Bike”` → `Fixing My Children's Bike`
- `'Don't Panic' Guide` → `Don't Panic Guide`
- `validate_title("Fixing My Children's Bike", …)` → valid
- `validate_title('Fixing "My" Bike', …)` → `"contains quotes"`

### TITLE-CASING: Acronyms and inner-capital words keep their spelling
**Applies to:** titler
**Verification:** test

Title Case is applied to ordinary words only. Words with an uppercase letter after the first
character, or with a digit, keep their spelling (SQL, iPhone, mRNA, v2). Small words (a, of,
vs, …) are lowercase except first and last.

**Acceptance criteria:**
- `iphone vs iPhone mRNA SQL v2 setup` → `Iphone vs iPhone mRNA SQL v2 Setup`

### TITLE-DECASE-ROUNDTRIP: Training titles drop Title Case and `clean_title` restores it
**Applies to:** titler
**Verification:** test

The v2 model trains on titles with the first letter of each ordinary word lowercased
(`decase_title`). `clean_title` on a decased title gives back the original, except for rare
Unicode cases (see the plan's round-trip count).

**Acceptance criteria:**
- `decase_title("Fixing My Children's Bike")` → `fixing my children's bike`
- `decase_title("iPhone SQL Setup Guide")` → `iPhone SQL setup guide`
- `clean_title(decase_title(t)) == t` for `Fixing My Children's Bike`, `Python vs Rust for CLI Tools`, `A Guide to mRNA Vaccines`

### TITLE-DECODE: The shipped model is decoded with 4-beam search, a title-only repeat ban and a filler penalty
**Applies to:** titler, app/core
**Verification:** test
**External tests:** `app/core/tests/golden.rs` (`beam_search_matches_huggingface`), cases made by `scripts/24_make_app_golden.py`

Titles come from beam search that follows Hugging Face `generate(num_beams=N)` (length penalty 1, no early stop): 4 beams by
default, at most 40 new tokens after `BOS + first 256 prompt tokens + <sep>`. Inside the title (not the prompt) a token pair is
never repeated and a word-start token never follows itself. The log-probability of "request" and "inquiry" is lowered by 3.0. The
Python code (`scripts/15_gen_v2_titles.py`) and the Rust code (`app/core`) give the same tokens.

**Acceptance criteria:**
- 24 toy-model cases (1, 2, 4 and 6 beams; ban on and off; penalty 0 and 3.0; stop by end token and by length) → Rust token ids equal the Hugging Face ids

### TITLE-TOKENIZE-PARITY: The Rust tokenizer equals the training tokenizer
**Applies to:** app/core
**Verification:** test
**External tests:** `app/core/tests/golden.rs` (`tokenizer_matches_python`)

The Rust tokenizer reads the vocabulary from the GGUF and gives the same ids as the Hugging Face tokenizer (`tokenizer.json`) for
any text, including text with characters outside the vocabulary (spelled out as byte tokens). A split of a long run such as
`____` may differ when both splits have the same score; the decoded text is then the same.

**Acceptance criteria:**
- 1,684 eval, non-English and edge-case texts → identical ids, except at most 3 equal-text ties
- `""` → no tokens

### TITLE-CLEAN-PARITY: Rust `clean_title` equals Python `clean_title`
**Applies to:** app/core
**Verification:** test
**External tests:** `app/core/tests/golden.rs` (`clean_title_matches_python`)

The Rust port gives the same output as `clean_title` (TITLE-PRODUCT-CLAMP, TITLE-APOSTROPHE, TITLE-CASING), except that it does
not strip teacher-only `<think>` tags.

**Acceptance criteria:**
- 1,533 decased training titles and 30 edge cases (quotes, markdown, leaked format ranges, line breaks) → identical strings
