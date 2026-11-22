"""Local correction for punctuation-count instructions in IFEval.

lm_eval 0.4.13 randomly replaces non-a-z `letter` values, including dataset
keys 1122 (#) and 1129 (!). Keep the literal character requested by the data.
This modifies the current process's registry only, never installed packages.
"""


def install_literal_character_checker():
    from lm_eval.tasks.ifeval import utils

    registry = utils.instructions_registry.INSTRUCTION_DICT
    original = registry["keywords:letter_frequency"]
    if getattr(original, "literal_character_fix", False):
        return

    class LiteralCharacterFrequencyChecker(original):
        literal_character_fix = True

        def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
            literal = isinstance(letter, str) and len(letter) == 1
            special = literal and not ("a" <= letter.lower() <= "z")
            description = super().build_description(
                letter="a" if special else letter,
                let_frequency=let_frequency, let_relation=let_relation,
            )
            if special:
                self._letter = letter.lower()
                description = self._description_pattern.format(
                    letter=self._letter, let_relation=self._comparison_relation,
                    let_frequency=self._frequency,
                )
            return description

    registry["keywords:letter_frequency"] = LiteralCharacterFrequencyChecker
