from models.workspace import PaperAnnotation, WorkspaceState
from services.paper_identity_service import (
    build_paper_identity_key,
    migrate_legacy_annotations,
    normalize_reader_lookup_url,
    resolve_identity_hints,
    upgrade_workspace_state,
)


def test_identity_key_prefers_semantic_scholar_id():
    key = build_paper_identity_key(
        "ABC123",
        "Memorability: How what we see influences what we remember",
        "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf",
    )
    assert key == "ssid:abc123"


def test_fallback_identity_key_is_deterministic():
    key_a = build_paper_identity_key(
        None,
        "A Paper Title",
        "https://example.org/paper.pdf#section-1",
    )
    key_b = build_paper_identity_key(
        "",
        "a paper title",
        "https://example.org/paper.pdf#another",
    )
    assert key_a.startswith("fallback:")
    assert key_a == key_b


def test_resolve_identity_hints_extracts_ssid_from_semantic_scholar_url():
    identity_key, ssid, normalized_title, normalized_url = resolve_identity_hints(
        annotation_key="https://www.semanticscholar.org/paper/Foo/186336411",
        paper_title="Memorability",
        source_url="https://www.semanticscholar.org/paper/Foo/186336411",
    )
    assert identity_key == "ssid:186336411"
    assert ssid == "186336411"
    assert normalized_title == "memorability"
    assert normalized_url == normalize_reader_lookup_url(
        "https://www.semanticscholar.org/paper/Foo/186336411"
    )


def test_migration_latest_note_wins_and_archives_others():
    legacy = {
        "Paper Alias A": PaperAnnotation(
            paperTitle="Paper Canonical",
            notesMarkdown="older note",
            sourceUrl="https://example.org/old",
            updatedAt="2024-01-01T00:00:00Z",
        ),
        "Paper Alias B": PaperAnnotation(
            paperTitle="Paper Canonical",
            notesMarkdown="newer note",
            sourceUrl="https://example.org/new",
            updatedAt="2024-06-01T00:00:00Z",
        ),
    }
    migrated, archive = migrate_legacy_annotations(
        legacy_annotations=legacy,
        now_iso="2026-01-01T00:00:00Z",
    )
    assert len(migrated) == 1
    only_annotation = list(migrated.values())[0]
    assert only_annotation.notesMarkdown == "newer note"
    assert sorted(only_annotation.migratedFromKeys) == ["Paper Alias A", "Paper Alias B"]
    assert len(archive) == 1
    assert archive[0].annotation.notesMarkdown == "older note"
    assert archive[0].originalKey == "Paper Alias A"


def test_upgrade_workspace_state_preserves_all_legacy_notes_without_loss():
    legacy_state = WorkspaceState(
        workspaceSchemaVersion=1,
        readingItems=[],
        themeNotes=[],
        paperAnnotations={
            "https://www.semanticscholar.org/paper/Foo/186336411": PaperAnnotation(
                paperTitle="Memorability",
                notesMarkdown="primary note",
                sourceUrl="https://www.semanticscholar.org/paper/Foo/186336411",
                updatedAt="2024-06-01T00:00:00Z",
            ),
            "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf": PaperAnnotation(
                paperTitle="Memorability",
                notesMarkdown="older url note",
                sourceUrl="https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf",
                updatedAt="2024-01-01T00:00:00Z",
            ),
        },
    )
    upgraded = upgrade_workspace_state(legacy_state, now_iso="2026-01-01T00:00:00Z")
    assert upgraded.workspaceSchemaVersion == 2
    assert len(upgraded.paperAnnotationsV2) == 1
    canonical = upgraded.paperAnnotationsV2["ssid:186336411"]
    assert canonical.notesMarkdown == "primary note"
    assert sorted(canonical.migratedFromKeys) == sorted(legacy_state.paperAnnotations.keys())
    assert len(upgraded.annotationMigrationArchive) == 1
    assert upgraded.annotationMigrationArchive[0].annotation.notesMarkdown == "older url note"


def test_migration_merges_publisher_alias_into_unique_ssid_group_by_title():
    legacy = {
        "https://www.semanticscholar.org/paper/Foo/186336411": PaperAnnotation(
            paperTitle="Memorability",
            notesMarkdown="semantic note",
            sourceUrl="https://www.semanticscholar.org/paper/Foo/186336411",
            updatedAt="2024-06-01T00:00:00Z",
        ),
        "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf": PaperAnnotation(
            paperTitle="Memorability",
            notesMarkdown="publisher note",
            sourceUrl="https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf",
            updatedAt="2024-01-01T00:00:00Z",
        ),
    }
    migrated, archive = migrate_legacy_annotations(
        legacy_annotations=legacy,
        now_iso="2026-01-01T00:00:00Z",
    )
    assert set(migrated.keys()) == {"ssid:186336411"}
    assert len(archive) == 1
    assert archive[0].annotation.notesMarkdown == "publisher note"


def test_migration_avoids_cross_paper_merge_when_title_maps_to_multiple_ssid_groups():
    legacy = {
        "https://www.semanticscholar.org/paper/Foo/111": PaperAnnotation(
            paperTitle="Shared Title",
            notesMarkdown="paper one",
            sourceUrl="https://www.semanticscholar.org/paper/Foo/111",
            updatedAt="2024-06-01T00:00:00Z",
        ),
        "https://www.semanticscholar.org/paper/Bar/222": PaperAnnotation(
            paperTitle="Shared Title",
            notesMarkdown="paper two",
            sourceUrl="https://www.semanticscholar.org/paper/Bar/222",
            updatedAt="2024-07-01T00:00:00Z",
        ),
        "https://publisher.example/shared-title.pdf": PaperAnnotation(
            paperTitle="Shared Title",
            notesMarkdown="publisher alias",
            sourceUrl="https://publisher.example/shared-title.pdf",
            updatedAt="2024-05-01T00:00:00Z",
        ),
    }
    migrated, archive = migrate_legacy_annotations(
        legacy_annotations=legacy,
        now_iso="2026-01-01T00:00:00Z",
    )
    assert "ssid:111" in migrated
    assert "ssid:222" in migrated
    fallback_keys = [key for key in migrated.keys() if key.startswith("fallback:")]
    assert len(fallback_keys) == 1
    assert migrated[fallback_keys[0]].notesMarkdown == "publisher alias"
    assert len(archive) == 0
