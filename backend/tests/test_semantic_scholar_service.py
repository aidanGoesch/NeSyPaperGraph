import services.semantic_scholar_service as semantic_module
from services.semantic_scholar_service import (
    SemanticScholarError,
    SemanticScholarService,
    ResolvedReference,
)


def test_resolve_reference_for_pmc_uses_idconv_identifiers(monkeypatch):
    service = SemanticScholarService(api_key="test-key")

    monkeypatch.setattr(
        service,
        "_resolve_pmc_identifiers",
        lambda pmc_id: ["DOI:10.1000/example", "PMID:12345678"]
        if pmc_id == "PMC11058347"
        else [],
    )

    ref = service._resolve_reference(
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11058347/",
        paper_id=None,
    )

    assert isinstance(ref, ResolvedReference)
    assert ref is not None
    assert ref.identifiers[:2] == ["DOI:10.1000/example", "PMID:12345678"]
    assert ref.identifiers[-1].startswith("URL:https://pmc.ncbi.nlm.nih.gov/articles/")


def test_hydrate_paper_tries_multiple_identifiers(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    calls = []

    monkeypatch.setattr(
        service,
        "_resolve_reference",
        lambda url, paper_id: ResolvedReference(
            identifiers=["DOI:missing", "PMID:1234"],
            source_url=url,
        ),
    )

    def _fake_request(path, query=None, suppress_not_found=False):
        _ = query
        _ = suppress_not_found
        calls.append(path)
        if path.endswith("DOI:missing"):
            return None
        return {
            "paperId": "paper-1234",
            "title": "NIH Paper",
            "authors": [{"name": "A. Author"}],
            "year": 2024,
            "venue": "Nature",
            "abstract": "Example abstract.",
            "url": "https://www.semanticscholar.org/paper/paper-1234",
        }

    monkeypatch.setattr(service, "_request_json", _fake_request)

    payload = service.hydrate_paper(
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC11058347/",
        paper_id=None,
    )

    assert payload is not None
    assert payload["semanticScholarPaperId"] == "paper-1234"
    assert payload["title"] == "NIH Paper"
    assert len(calls) == 2


def test_compute_centroid_embedding_averages_vectors():
    service = SemanticScholarService(api_key="test-key")
    centroid = service.compute_centroid_embedding(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
    )
    assert centroid == [3.0, 4.0, 5.0]


def test_rank_candidates_by_embedding_similarity_orders_descending():
    service = SemanticScholarService(api_key="test-key")
    profile_embedding = [1.0, 0.0]
    candidates = [
        {
            "paperId": "aligned",
            "title": "Aligned",
            "embedding": [0.9, 0.1],
        },
        {
            "paperId": "orthogonal",
            "title": "Orthogonal",
            "embedding": [0.0, 1.0],
        },
        {
            "paperId": "inverse",
            "title": "Inverse",
            "embedding": [-1.0, 0.0],
        },
    ]
    ranked = service.rank_candidates_by_embedding_similarity(
        profile_embedding=profile_embedding,
        candidates=candidates,
        limit=2,
    )
    assert [item["paperId"] for item in ranked] == ["aligned", "orthogonal"]
    assert ranked[0]["score"] > ranked[1]["score"]


def test_resolve_seed_paper_details_falls_back_to_title_search(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(
        service,
        "search_papers",
        lambda query, limit=20: [
            {"paperId": "paper-1", "title": "Almost Match", "authors": [], "year": 2020},
            {"paperId": "paper-2", "title": "Exact Match Paper", "authors": [], "year": 2024},
        ]
        if query == "Exact Match Paper"
        else [],
    )
    monkeypatch.setattr(
        service,
        "fetch_paper_details",
        lambda paper_id: {"paperId": paper_id, "embedding": {"vector": [0.1, 0.2]}}
        if paper_id in {"paper-1", "paper-2"}
        else None,
    )
    details = service.resolve_seed_paper_details({"title": "Exact Match Paper"})
    assert details is not None
    assert details["paperId"] == "paper-2"


def test_find_similar_papers_uses_fallback_queries_when_primary_returns_only_seed(
    monkeypatch,
):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(
        service,
        "fetch_paper_details",
        lambda paper_id: {
            "paperId": paper_id,
            "title": "Very Specific Seed Paper Title",
            "authors": ["Author One"],
            "year": 2023,
            "abstract": "This abstract has many unique terms.",
            "embedding": None,
        },
    )
    search_calls = []

    def _fake_search(query, limit=20):
        _ = limit
        search_calls.append(query)
        if len(search_calls) == 1:
            return [{"paperId": "seed-1", "title": "Very Specific Seed Paper Title"}]
        return [{"paperId": "rec-1", "title": "Fallback Recommendation"}]

    monkeypatch.setattr(
        service,
        "fetch_recommended_papers_for_paper",
        lambda paper_id, limit=20: [],
    )
    monkeypatch.setattr(service, "search_papers", _fake_search)

    results = service.find_similar_papers("seed-1", limit=5)

    assert len(search_calls) >= 2
    assert results
    assert results[0]["paperId"] == "rec-1"


def test_find_similar_papers_prefers_recommendations_endpoint(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(
        service,
        "fetch_paper_details",
        lambda paper_id: {
            "paperId": paper_id,
            "title": "Seed Paper",
            "authors": ["Author One"],
            "year": 2024,
            "abstract": "Seed abstract.",
            "embedding": None,
        },
    )
    monkeypatch.setattr(
        service,
        "fetch_recommended_papers_for_paper",
        lambda paper_id, limit=20: [
            {"paperId": "rec-1", "title": "Recommended Paper One"},
            {"paperId": paper_id, "title": "Seed Paper"},
        ],
    )
    monkeypatch.setattr(
        service,
        "search_papers",
        lambda query, limit=20: [
            {"paperId": "search-1", "title": "Search Candidate"},
        ],
    )

    results = service.find_similar_papers("seed-1", limit=5)

    assert results
    assert results[0]["paperId"] == "rec-1"
    assert all(item["paperId"] != "seed-1" for item in results)


def test_recommend_for_theme_uses_fallback_queries_when_primary_empty(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(service, "_build_profile_embedding", lambda seed_papers: [])
    search_calls = []

    def _fake_search(query, limit=20):
        _ = limit
        search_calls.append(query)
        if len(search_calls) == 1:
            return []
        return [{"paperId": "theme-rec-1", "title": "Fallback Theme Candidate"}]

    monkeypatch.setattr(service, "search_papers", _fake_search)

    results = service.recommend_for_theme(
        theme_title="Causal Memory",
        seed_papers=[{"title": "Seed Paper A", "topicHints": ["memory"]}],
        limit=3,
        candidate_pool_size=8,
    )

    assert len(search_calls) >= 2
    assert results
    assert results[0]["paperId"] == "theme-rec-1"


def test_build_profile_embedding_skips_seed_errors(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    seeds = [{"title": "first"}, {"title": "second"}]
    calls = {"count": 0}

    def _resolve(seed):
        calls["count"] += 1
        if calls["count"] == 1:
            raise SemanticScholarError("rate limited")
        return {"paperId": "ok-1", "embedding": {"vector": [0.2, 0.4]}}

    monkeypatch.setattr(service, "resolve_seed_paper_details", _resolve)
    embedding = service._build_profile_embedding(seeds)
    assert embedding == [0.2, 0.4]


def test_recommend_for_theme_prefers_seed_neighborhood_candidates(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(service, "_build_profile_embedding", lambda seed_papers: [])
    monkeypatch.setattr(
        service,
        "resolve_seed_paper_details",
        lambda seed: {"paperId": "seed-1", "title": "Seed Title", "embedding": None},
    )
    monkeypatch.setattr(
        service,
        "fetch_recommended_papers_for_paper",
        lambda paper_id, limit=20: [
            {"paperId": "seed-1", "title": "Seed Title"},
            {"paperId": "rec-from-seed", "title": "Seed Neighborhood Candidate"},
        ],
    )
    monkeypatch.setattr(
        service,
        "search_papers",
        lambda query, limit=20: [{"paperId": "rec-from-search", "title": "Fallback Search Candidate"}],
    )

    results = service.recommend_for_theme(
        theme_title="Theme Title",
        seed_papers=[{"title": "Seed Title"}],
        limit=2,
        candidate_pool_size=2,
    )

    assert results
    assert results[0]["paperId"] == "rec-from-seed"


def test_compute_backoff_seconds_exponential_and_capped(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    monkeypatch.setattr(semantic_module, "SEMANTIC_SCHOLAR_BACKOFF_BASE_SECONDS", 1.0)
    monkeypatch.setattr(semantic_module, "SEMANTIC_SCHOLAR_BACKOFF_MAX_SECONDS", 3.0)
    assert service._compute_backoff_seconds(0) == 1.0
    assert service._compute_backoff_seconds(1) == 2.0
    assert service._compute_backoff_seconds(3) == 3.0


def test_wait_for_rate_limit_slot_sleeps_when_called_too_soon(monkeypatch):
    service = SemanticScholarService(api_key="test-key")
    service._last_request_monotonic = 10.0
    monkeypatch.setattr(semantic_module, "SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS", 1.0)
    monotonic_values = iter([10.2, 11.3])
    sleep_calls = []

    monkeypatch.setattr(
        semantic_module.time, "monotonic", lambda: next(monotonic_values)
    )
    monkeypatch.setattr(semantic_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    service._wait_for_rate_limit_slot()

    assert sleep_calls
    assert round(sleep_calls[0], 2) == 0.8
