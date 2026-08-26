from __future__ import annotations

import unittest

from baseline.interpreter import ParseResult, parse


def rows(result: ParseResult) -> list[tuple[str, list[str], str, str, str, str]]:
    return [
        (item.attribute, item.values, item.relation, item.polarity, item.strength, item.operator)
        for item in result.constraints
    ]


class InterpreterTest(unittest.TestCase):
    def test_public_templates(self) -> None:
        result = parse("I'm looking for trail shoes. A key requirement is: black leather.", 2)
        self.assertEqual([item.attribute for item in result.constraints], ["category", "color", "material"])
        self.assertEqual(result.constraints[0].values, ["trail shoes"])
        self.assertEqual(result.constraints[1].values, ["black"])
        self.assertEqual(result.constraints[2].values, ["leather"])
        self.assertTrue(all(item.strength == "hard" for item in result.constraints[1:]))
        self.assertEqual(result.constraints[0].source_text, "I'm looking for trail shoes. A key requirement is: black leather.")

        answer = parse("For that, what matters is: a; b.", 3, "color")
        self.assertEqual(rows(answer), [("color", ["a", "b"], "any", "include", "unspecified", "none")])
        self.assertEqual(answer.constraints[0].provenance, "simulator_answer")

        known_answer = parse("For that, what matters is: navy.", 3, "other")
        self.assertEqual(known_answer.constraints[0].attribute, "color")
        self.assertEqual(known_answer.constraints[0].provenance, "simulator_answer")

        joined_answer = parse(
            "For that, what matters is: machine washable and wrinkle resistant.",
            3,
            "feature",
        )
        self.assertEqual(joined_answer.constraints[0].values, ["machine washable", "wrinkle resistant"])
        self.assertEqual(joined_answer.constraints[0].relation, "all")

        browsing = parse("I'm looking for shirts, but I'm still exploring.", 1)
        self.assertEqual(rows(browsing), [("category", ["shirts"], "any", "include", "unspecified", "none")])

    def test_no_preference_and_targeted_override(self) -> None:
        no_preference = parse("I don't have an additional preference for material.", 4)
        self.assertEqual(no_preference.no_preference_attributes, frozenset({"material"}))
        self.assertEqual(no_preference.constraints, ())

        override = parse("Actually, ignore my earlier preference. What I need is: navy.", 4)
        self.assertEqual(override.correction_attributes, frozenset({"color"}))
        self.assertEqual(rows(override), [("color", ["navy"], "any", "include", "hard", "none")])

    def test_conjunction_negation_and_independent_values(self) -> None:
        cases = (
            ("black or navy", ("color", ["black", "navy"], "any", "include")),
            ("waterproof and lightweight", ("feature", ["waterproof", "lightweight"], "all", "include")),
            ("not leather", ("material", ["leather"], "any", "exclude")),
            ("avoid leather", ("material", ["leather"], "any", "exclude")),
            ("anything but leather", ("material", ["leather"], "any", "exclude")),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                item = parse(message, 1).constraints[0]
                self.assertEqual((item.attribute, item.values, item.relation, item.polarity), expected)

        mixed = parse("black leather", 2)
        self.assertEqual([(item.attribute, item.values) for item in mixed.constraints], [("color", ["black"]), ("material", ["leather"])])

    def test_strength_and_budget_operators(self) -> None:
        strength_cases = (
            ("It must be waterproof", "hard"),
            ("I prefer black", "soft"),
            ("It would be nice to have lightweight", "soft"),
        )
        for message, expected in strength_cases:
            with self.subTest(message=message):
                self.assertEqual(parse(message, 1).constraints[0].strength, expected)

        budget_cases = (
            ("under $80", "80", "at_most"),
            ("at most $80", "80", "at_most"),
            ("over $20", "20", "at_least"),
            ("at least $20", "20", "at_least"),
            ("exactly $50", "50", "equals"),
            ("budget around $19.99", "19.99", "equals"),
        )
        for message, value, operator in budget_cases:
            with self.subTest(message=message):
                item = parse(message, 1).constraints[0]
                self.assertEqual((item.attribute, item.values, item.operator), ("budget", [value], operator))
                self.assertEqual(len(parse(message, 1).constraints), 1)
                if operator != "equals":
                    self.assertEqual(item.strength, "hard")

    def test_ambiguous_answer_does_not_guess(self) -> None:
        result = parse("For that, what matters is: something.", 2, "color")
        self.assertEqual(result.constraints, ())
        self.assertEqual(result.correction_attributes, frozenset())

    def test_explicit_unknown_public_requirement_is_retained(self) -> None:
        result = parse(
            "I'm looking for jackets. A key requirement is: machine washable.",
            1,
        )

        self.assertEqual(
            [(item.attribute, item.values) for item in result.constraints],
            [("category", ["jackets"]), ("feature", ["machine washable"])],
        )


if __name__ == "__main__":
    unittest.main()
