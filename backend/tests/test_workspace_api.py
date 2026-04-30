def _workspace_payload():
    return {
        "readingItems": [
            {
                "id": "read-1",
                "sourceType": "url",
                "status": "inbox",
                "topicHints": ["NLP"],
                "linkedPaperTitle": None,
                "linkedThemeId": None,
                "title": "Interesting Read",
                "url": "https://example.com",
                "semanticScholarPaperId": "ss-abc123",
                "authors": ["First Author", "Second Author"],
                "year": 2024,
                "venue": "ACL",
                "quickNote": "note",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        ],
        "themeNotes": [
            {
                "id": "theme-1",
                "themeTitle": "Transformers",
                "linkedPaperTitles": [],
                "sections": {"notes": "n", "toRead": "r"},
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        ],
        "paperAnnotations": {
            "Paper A": {
                "paperTitle": "Paper A",
                "notesMarkdown": "x",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        },
    }


def test_workspace_get_default(client, monkeypatch):
    monkeypatch.setattr("api.workspace.load_workspace_state", lambda: None)
    response = client.get("/api/workspace/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["readingItems"] == []
    assert payload["themeNotes"] == []
    assert payload["paperAnnotations"] == {}


def test_workspace_put_and_get_roundtrip(client, monkeypatch):
    stored = {}

    def _load():
        return stored.get("state")

    def _save(state):
        stored["state"] = state

    monkeypatch.setattr("api.workspace.load_workspace_state", _load)
    monkeypatch.setattr("api.workspace.save_workspace_state", _save)
    monkeypatch.setattr("api.workspace.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    payload = _workspace_payload()
    put_response = client.put("/api/workspace/state", json=payload)
    assert put_response.status_code == 200
    put_payload = put_response.json()
    assert put_payload["readingItems"][0]["updatedAt"] == "2026-01-01T00:00:00Z"

    get_response = client.get("/api/workspace/state")
    assert get_response.status_code == 200
    assert get_response.json()["readingItems"][0]["id"] == "read-1"


def test_workspace_put_validation_error(client):
    bad_payload = {
        "readingItems": [],
        "themeNotes": [
            {
                "id": "theme-1",
                "themeTitle": "   ",
                "linkedPaperTitles": [],
                "sections": {"notes": "", "toRead": ""},
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:00:00Z",
            }
        ],
        "paperAnnotations": {},
    }
    response = client.put("/api/workspace/state", json=bad_payload)
    assert response.status_code == 422


def test_workspace_put_accepts_legacy_items_without_semantic_scholar_fields(
    client, monkeypatch
):
    stored = {}

    def _load():
        return stored.get("state")

    def _save(state):
        stored["state"] = state

    monkeypatch.setattr("api.workspace.load_workspace_state", _load)
    monkeypatch.setattr("api.workspace.save_workspace_state", _save)
    monkeypatch.setattr("api.workspace.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    payload = _workspace_payload()
    payload["readingItems"][0].pop("semanticScholarPaperId", None)
    payload["readingItems"][0].pop("authors", None)
    payload["readingItems"][0].pop("year", None)
    payload["readingItems"][0].pop("venue", None)

    response = client.put("/api/workspace/state", json=payload)
    assert response.status_code == 200
