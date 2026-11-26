from src.core import chunk_documents, is_hit, get_chunk_map, local_rewrites


def test_chunk_documents_overlap_and_ids():
    docs = [{"doc_id": "policy", "source": "policy.txt", "text": "A " * 100}]
    chunks = chunk_documents(docs, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "policy__000"


def test_is_hit_checks_expected_phrase():
    chunks = [{"chunk_id": "c1", "text": "Casual leave is 12 days."}]
    assert is_hit(["c1"], get_chunk_map(chunks), ["casual leave"])
    assert not is_hit(["c1"], get_chunk_map(chunks), ["sick leave"])


def test_local_rewrites_returns_three_queries():
    rewrites = local_rewrites("WFH rules?")
    assert len(rewrites) == 3
    assert any("work from home" in q.lower() for q in rewrites)
