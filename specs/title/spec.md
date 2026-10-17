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
