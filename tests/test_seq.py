import pytest
from oritatami.seq import (
    SequenceError,
    apply_mutations,
    clean_sequence,
    diff_substitutions,
    identity,
    parse_fasta,
    parse_mutations,
)

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def test_clean_sequence_strips_whitespace_and_digits():
    assert clean_sequence(" mqif 10 vk\n", "protein") == "MQIFVK"


def test_clean_sequence_rejects_bad_letters():
    with pytest.raises(SequenceError, match="B"):
        clean_sequence("MQBF", "protein")
    with pytest.raises(SequenceError):
        clean_sequence("ACGU", "dna")
    assert clean_sequence("acgu", "rna") == "ACGU"


def test_parse_fasta_multi_and_bare():
    recs = parse_fasta(">a desc\nMQ\nIF\n>b\nKK\n")
    assert recs == [("a desc", "MQIF"), ("b", "KK")]
    assert parse_fasta("MQIF") == [("sequence", "MQIF")]


def test_parse_and_apply_mutations():
    muts = parse_mutations("K48R, A:K63R")
    assert [m.code for m in muts] == ["K48R", "A:K63R"]
    new = apply_mutations(UBQ, muts)
    assert new[47] == "R" and new[62] == "R"
    assert diff_substitutions(UBQ, new) == ["K48R", "K63R"]


def test_apply_mutations_checks_wild_type():
    with pytest.raises(SequenceError, match="位置 48 の残基は K"):
        apply_mutations(UBQ, parse_mutations("A48R"))
    with pytest.raises(SequenceError, match="範囲外"):
        apply_mutations(UBQ, parse_mutations("K480R"))


def test_parse_mutations_rejects_garbage():
    with pytest.raises(SequenceError):
        parse_mutations("K48")
    with pytest.raises(SequenceError):
        parse_mutations("B48R")


def test_identity():
    assert identity(UBQ, UBQ) == 1.0
    assert 0.95 < identity(UBQ, apply_mutations(UBQ, parse_mutations("K48R K63R"))) < 1.0


def test_pasted_fasta_is_accepted():
    """Copying from UniProt brings the header line; that is the normal case, not junk."""
    from oritatami.seq import SequenceError, clean_sequence

    UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    assert clean_sequence(">sp|P0CG48|UBC_HUMAN Polyubiquitin-C\n" + UBQ) == UBQ
    # wrapped at 60 columns, the way every database serves it
    wrapped = ">x\n" + "\n".join(UBQ[i:i + 60] for i in range(0, len(UBQ), 60))
    assert clean_sequence(wrapped) == UBQ
    # the older dialect uses ; for comments
    assert clean_sequence("; a comment\n" + UBQ) == UBQ
    # a bare sequence still works, and lowercase is still normalised
    assert clean_sequence(UBQ.lower()) == UBQ

    # two records in a one-sequence field is a mistake worth naming, not concatenating
    with pytest.raises(SequenceError) as e:
        clean_sequence(f">a\n{UBQ}\n>b\n{UBQ}")
    assert "2 本" in str(e.value)

    with pytest.raises(SequenceError):
        clean_sequence(">header only\n")
