from services.semantic_scholar_service import SemanticScholarService, ResolvedReference


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
