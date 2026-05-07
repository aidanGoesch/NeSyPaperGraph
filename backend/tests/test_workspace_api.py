def _workspace_payload():
    return {
        "workspaceSchemaVersion": 2,
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
        "paperAnnotationsV2": {},
        "annotationMigrationArchive": [],
    }


def test_workspace_get_default(client, monkeypatch):
    monkeypatch.setattr("api.workspace.load_workspace_state", lambda: None)
    response = client.get("/api/workspace/state")
    assert response.status_code == 200
    payload = response.json()
    assert payload["workspaceSchemaVersion"] == 2
    assert payload["readingItems"] == []
    assert payload["themeNotes"] == []
    assert payload["paperAnnotations"] == {}
    assert payload["paperAnnotationsV2"] == {}
    assert payload["annotationMigrationArchive"] == []


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
    assert put_payload["workspaceSchemaVersion"] == 2
    assert put_payload["readingItems"][0]["updatedAt"] == "2026-01-01T00:00:00Z"
    assert len(put_payload["paperAnnotationsV2"]) == 1

    get_response = client.get("/api/workspace/state")
    assert get_response.status_code == 200
    assert get_response.json()["readingItems"][0]["id"] == "read-1"
    assert len(get_response.json()["paperAnnotationsV2"]) == 1


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
        "paperAnnotationsV2": {},
        "annotationMigrationArchive": [],
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
    body = response.json()
    assert body["workspaceSchemaVersion"] == 2
    assert len(body["paperAnnotationsV2"]) == 1


def test_workspace_put_non_destructive_v2_merge(client, monkeypatch):
    stored = {}

    def _load():
        return stored.get("state")

    def _save(state):
        stored["state"] = state

    monkeypatch.setattr("api.workspace.load_workspace_state", _load)
    monkeypatch.setattr("api.workspace.save_workspace_state", _save)
    monkeypatch.setattr("api.workspace.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    initial_payload = _workspace_payload()
    first_response = client.put("/api/workspace/state", json=initial_payload)
    assert first_response.status_code == 200
    first_body = first_response.json()
    v2_entries = first_body["paperAnnotationsV2"]
    first_key = next(iter(v2_entries.keys()))
    first_entry = v2_entries[first_key]

    second_identity = {
        "paperIdentityKey": "ssid:second-paper",
        "paperTitle": "Paper B",
        "semanticScholarPaperId": "second-paper",
        "normalizedTitle": "paper b",
        "canonicalSourceUrl": "https://example.org/second",
        "sourceUrlAliases": ["https://example.org/second"],
        "notesMarkdown": "paper b note",
        "topicLinks": [],
        "status": "unread",
        "migratedFromKeys": ["Paper B"],
        "selectedPrimarySource": "Paper B",
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
    }

    update_payload = _workspace_payload()
    update_payload["paperAnnotations"] = {}
    update_payload["paperAnnotationsV2"] = {
        first_key: {**first_entry, "notesMarkdown": "updated note"},
        "ssid:second-paper": second_identity,
    }
    update_payload["annotationMigrationArchive"] = []

    second_response = client.put("/api/workspace/state", json=update_payload)
    assert second_response.status_code == 200
    second_body = second_response.json()
    assert len(second_body["paperAnnotationsV2"]) == 2
    assert second_body["paperAnnotationsV2"][first_key]["notesMarkdown"] == "updated note"
    assert second_body["paperAnnotationsV2"]["ssid:second-paper"]["notesMarkdown"] == "paper b note"

    partial_payload = _workspace_payload()
    partial_payload["paperAnnotations"] = {}
    partial_payload["paperAnnotationsV2"] = {
        first_key: {**first_entry, "notesMarkdown": "partial update note"}
    }
    partial_payload["annotationMigrationArchive"] = []

    partial_response = client.put("/api/workspace/state", json=partial_payload)
    assert partial_response.status_code == 200
    partial_body = partial_response.json()
    assert len(partial_body["paperAnnotationsV2"]) == 2
    assert (
        partial_body["paperAnnotationsV2"]["ssid:second-paper"]["notesMarkdown"]
        == "paper b note"
    )
    assert partial_body["paperAnnotationsV2"][first_key]["notesMarkdown"] == "partial update note"


def test_workspace_put_rejects_v2_identity_key_mismatch(client, monkeypatch):
    monkeypatch.setattr("api.workspace.load_workspace_state", lambda: None)
    monkeypatch.setattr("api.workspace.save_workspace_state", lambda _state: None)

    payload = _workspace_payload()
    payload["paperAnnotations"] = {}
    payload["paperAnnotationsV2"] = {
        "ssid:paper-a": {
            "paperIdentityKey": "ssid:different",
            "paperTitle": "Paper A",
            "semanticScholarPaperId": "paper-a",
            "normalizedTitle": "paper a",
            "canonicalSourceUrl": "https://example.org/a",
            "sourceUrlAliases": ["https://example.org/a"],
            "notesMarkdown": "x",
            "topicLinks": [],
            "status": "unread",
            "migratedFromKeys": ["Paper A"],
            "selectedPrimarySource": "Paper A",
            "createdAt": "2026-01-01T00:00:00Z",
            "updatedAt": "2026-01-01T00:00:00Z",
        }
    }
    payload["annotationMigrationArchive"] = []

    response = client.put("/api/workspace/state", json=payload)
    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]


def test_workspace_sequential_link_navigation_keeps_same_paper_identity(client, monkeypatch):
    stored = {}

    def _load():
        return stored.get("state")

    def _save(state):
        stored["state"] = state

    monkeypatch.setattr("api.workspace.load_workspace_state", _load)
    monkeypatch.setattr("api.workspace.save_workspace_state", _save)
    monkeypatch.setattr("api.workspace.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    semantic_payload = {
        "workspaceSchemaVersion": 1,
        "readingItems": [],
        "themeNotes": [],
        "paperAnnotations": {
            "https://www.semanticscholar.org/paper/Foo/186336411": {
                "paperTitle": "Memorability",
                "notesMarkdown": "note from semantic scholar",
                "sourceUrl": "https://www.semanticscholar.org/paper/Foo/186336411",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2024-06-01T00:00:00Z",
            }
        },
    }
    first_response = client.put("/api/workspace/state", json=semantic_payload)
    assert first_response.status_code == 200
    first_body = first_response.json()
    assert set(first_body["paperAnnotationsV2"].keys()) == {"ssid:186336411"}

    publisher_payload = {
        "workspaceSchemaVersion": 1,
        "readingItems": [],
        "themeNotes": [],
        "paperAnnotations": {
            "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf": {
                "paperTitle": "Memorability",
                "notesMarkdown": "note from publisher link",
                "sourceUrl": "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2024-07-01T00:00:00Z",
            }
        },
    }
    second_response = client.put("/api/workspace/state", json=publisher_payload)
    assert second_response.status_code == 200
    second_body = second_response.json()
    assert set(second_body["paperAnnotationsV2"].keys()) == {"ssid:186336411"}
    assert (
        second_body["paperAnnotationsV2"]["ssid:186336411"]["notesMarkdown"]
        == "note from publisher link"
    )
    assert len(second_body["annotationMigrationArchive"]) >= 1


def test_workspace_same_title_different_ssid_notes_do_not_bleed(client, monkeypatch):
    stored = {}

    def _load():
        return stored.get("state")

    def _save(state):
        stored["state"] = state

    monkeypatch.setattr("api.workspace.load_workspace_state", _load)
    monkeypatch.setattr("api.workspace.save_workspace_state", _save)
    monkeypatch.setattr("api.workspace.utc_now_iso", lambda: "2026-01-01T00:00:00Z")

    payload = {
        "workspaceSchemaVersion": 1,
        "readingItems": [],
        "themeNotes": [],
        "paperAnnotations": {
            "https://www.semanticscholar.org/paper/Shared/111": {
                "paperTitle": "Shared Title",
                "notesMarkdown": "note one",
                "sourceUrl": "https://www.semanticscholar.org/paper/Shared/111",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2024-06-01T00:00:00Z",
            },
            "https://www.semanticscholar.org/paper/Shared/222": {
                "paperTitle": "Shared Title",
                "notesMarkdown": "note two",
                "sourceUrl": "https://www.semanticscholar.org/paper/Shared/222",
                "topicLinks": [],
                "status": "unread",
                "updatedAt": "2024-07-01T00:00:00Z",
            },
        },
    }
    response = client.put("/api/workspace/state", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body["paperAnnotationsV2"].keys()) == {"ssid:111", "ssid:222"}
    assert body["paperAnnotationsV2"]["ssid:111"]["notesMarkdown"] == "note one"
    assert body["paperAnnotationsV2"]["ssid:222"]["notesMarkdown"] == "note two"
