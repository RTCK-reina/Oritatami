"""A proposal whose residue is right and whose index is wrong should be moved, not dropped.

Observed on PCNA: of three mutation proposals the model made, each carried one mutation that
applied and one that was rejected with "位置 257 の残基は A で、S ではありません". The
rejected halves were not nonsense — the residue named was real and a few positions away.
"""

import pytest
from oritatami import assistant
from oritatami.seq import Mutation, infer_offset, parse_mutations, repair_mutation, repair_mutations

# M1 K2 A3 S4 Y5 L6 L7 N8 Q9 V10 G11 W12 S13 T14 K15 A16 C17 ... 57 residues.
# D appears only at 18 and 38, which is what the "too far to be the same mutation" cases use.
PROTEIN = "MKASYLLNQVGWSTKACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYAC"


def test_a_correct_mutation_is_left_alone():
    mut = Mutation(wt="M", position=1, mt="A")
    moved, note = repair_mutation(PROTEIN, mut)
    assert moved == mut and note is None


def test_a_nearby_residue_is_used_and_the_move_is_reported():
    # position 3 is A; the nearest S is at 4
    mut = Mutation(wt="S", position=3, mt="P")
    moved, note = repair_mutation(PROTEIN, mut)
    assert moved is not None and moved.position == 4 and moved.mt == "P"
    assert "S3P → S4P" in note and "位置 3 は A" in note


def test_nothing_within_the_window_is_dropped_with_a_reason():
    mut = Mutation(wt="D", position=1, mt="A")      # D is only at 18 and 38
    moved, note = repair_mutation(PROTEIN, mut)
    assert moved is None and "D がありません" in note


def test_a_position_past_the_end_is_dropped():
    mut = Mutation(wt="K", position=500, mt="A")
    moved, note = repair_mutation(PROTEIN, mut)
    assert moved is None and "外" in note


def test_a_shared_offset_is_preferred_over_the_nearest_match():
    """Numbering a construct from the UniProt entry shifts every position by the same amount."""
    seq = PROTEIN
    shifted = [Mutation(wt=seq[i - 1], position=i + 4, mt="A") for i in (10, 20, 30)]
    assert infer_offset(seq, shifted) == -4
    kept, notes, dropped = repair_mutations(seq, shifted)
    assert [m.position for m in kept] == [10, 20, 30]
    assert dropped == [] and all("-4 ずれていました" in n for n in notes)


def test_one_wrong_mutation_alone_is_not_treated_as_an_offset():
    assert infer_offset(PROTEIN, [Mutation(wt="S", position=3, mt="P")]) is None


def test_two_repairs_landing_on_one_position_are_merged_and_said_so():
    seq = "MAAASAAAA"
    muts = parse_mutations("S3P, S7P")          # both repair onto position 5
    kept, notes, dropped = repair_mutations(seq, muts)
    assert [m.code for m in kept] == ["S5P"]
    assert dropped == [] and any("まとめました" in n for n in notes)


def test_two_repairs_onto_one_position_with_different_targets_drop_the_second():
    seq = "MAAASAAAA"
    kept, notes, dropped = repair_mutations(seq, parse_mutations("S3P, S7W"))
    assert [m.code for m in kept] == ["S5P"]
    assert any("重なります" in d for d in dropped)


@pytest.fixture
def workbench():
    return {"components": [{"type": "protein", "chains": ["A"], "sequence": PROTEIN, "label": "test"}]}


def test_the_proposal_survives_with_the_repair_named(workbench):
    out = assistant.verify_proposal(
        {"type": "mutation_set", "title": "t", "rationale": "r", "chain": "A",
         "mutations": ["M1A", "S3P"]}, workbench, "A")
    assert out["status"] == "warning"            # usable, but not what the model wrote
    assert out["mutations"] == ["M1A", "S4P"]
    assert out["apply"]["mutations"] == ["M1A", "S4P"]
    assert out["repaired"] and "S3P → S4P" in out["repaired"][0]
    assert out["issues"] == []                   # a repair is not a problem


def test_an_unplaceable_mutation_is_still_reported_and_the_rest_runs(workbench):
    out = assistant.verify_proposal(
        {"type": "mutation_set", "title": "t", "rationale": "r", "chain": "A",
         "mutations": ["M1A", "D2A"]}, workbench, "A")
    assert out["mutations"] == ["M1A"]
    assert out["issues"] and "D2A" in out["issues"][0]


def test_a_proposal_with_nothing_placeable_is_rejected(workbench):
    out = assistant.verify_proposal(
        {"type": "mutation_set", "title": "t", "rationale": "r", "chain": "A",
         "mutations": ["D2A"]}, workbench, "A")
    assert out["status"] == "invalid"


def test_the_rejection_message_does_not_repeat_the_code(workbench):
    """It used to read "S257P: S257P: 位置 257 の…" because both halves added the code."""
    out = assistant.verify_proposal(
        {"type": "mutation_set", "title": "t", "rationale": "r", "chain": "A",
         "mutations": ["M1A", "D2A"]}, workbench, "A")
    assert out["issues"][0].count("D2A") == 1
