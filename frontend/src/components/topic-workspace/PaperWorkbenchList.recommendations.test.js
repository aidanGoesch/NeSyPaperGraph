import React, { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import PaperWorkbenchList from "./PaperWorkbenchList";

jest.mock("./DesktopPaperWebview", () => (props) => (
    <div data-testid="desktop-paper-webview">{props.url}</div>
));

function renderList(overrides = {}) {
    const props = {
        papers: [
            {
                title: "Neuro-Symbolic Program Synthesis",
                authors: ["A. Author"],
                publication_date: "2024",
                abstract: "Combines neural and symbolic systems.",
                topics: ["Neurosymbolic AI"],
                url: "https://example.org/paper/neuro-symbolic",
            },
        ],
        totalPaperCount: 1,
        hasMorePapers: false,
        onLoadMorePapers: jest.fn(),
        selectedTopic: null,
        selectedTopicLabel: null,
        hasActiveFilter: false,
        onClearFilters: jest.fn(),
        onFocusPaper: jest.fn(),
        onOpenThemeAssignmentModal: jest.fn(),
        getPaperAnnotation: jest.fn(() => null),
        onUpdatePaperAnnotation: jest.fn(),
        requestedPaperTitle: null,
        onRequestSimilarPapers: jest.fn().mockResolvedValue([]),
        ...overrides,
    };
    render(<PaperWorkbenchList {...props} />);
    return props;
}

function renderListWithAnnotationState(overrides = {}) {
    const props = {
        papers: [
            {
                title: "Neuro-Symbolic Program Synthesis",
                authors: ["A. Author"],
                publication_date: "2024",
                abstract: "Combines neural and symbolic systems.",
                topics: ["Neurosymbolic AI"],
            },
        ],
        onResolvePaperMetadata: jest.fn(),
        ...overrides,
    };

    function Wrapper() {
        const [annotations, setAnnotations] = useState({});
        return (
            <PaperWorkbenchList
                papers={props.papers}
                totalPaperCount={props.papers.length}
                hasMorePapers={false}
                onLoadMorePapers={jest.fn()}
                selectedTopic={null}
                selectedTopicLabel={null}
                hasActiveFilter={false}
                onClearFilters={jest.fn()}
                onOpenThemeAssignmentModal={jest.fn()}
                getPaperAnnotation={(paperTitle) =>
                    annotations[paperTitle] || {
                        paperTitle,
                        notesMarkdown: "",
                        sourceUrl: "",
                        topicLinks: [],
                        status: "unread",
                    }
                }
                onUpdatePaperAnnotation={(paperTitle, patch) => {
                    setAnnotations((previous) => ({
                        ...previous,
                        [paperTitle]: {
                            paperTitle,
                            notesMarkdown: "",
                            sourceUrl: "",
                            topicLinks: [],
                            status: "unread",
                            ...(previous[paperTitle] || {}),
                            ...patch,
                        },
                    }));
                }}
                onRequestSimilarPapers={jest.fn().mockResolvedValue([])}
                onResolvePaperMetadata={props.onResolvePaperMetadata}
            />
        );
    }

    render(<Wrapper />);
    return props;
}

describe("PaperWorkbenchList recommendations", () => {
    beforeEach(() => {
        window.localStorage.removeItem("nesy_paper_notes_rich_editor_enabled");
    });

    test("selects requested paper using normalized title matching", async () => {
        renderList({
            requestedPaperTitle: "  neuro-symbolic program synthesis  ",
        });

        await waitFor(() =>
            expect(
                screen.getByRole("heading", {
                    level: 4,
                    name: "Neuro-Symbolic Program Synthesis",
                })
            ).toBeTruthy()
        );
    });

    test("fetches similar papers for selected paper", async () => {
        const onRequestSimilarPapers = jest.fn().mockResolvedValue([
            { paperId: "rec-1", title: "Composable Neuro-Symbolic Inference" },
        ]);
        renderList({ onRequestSimilarPapers });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getAllByText("See similar papers")[0]);

        await waitFor(() =>
            expect(onRequestSimilarPapers).toHaveBeenCalledWith(
                expect.objectContaining({
                    title: "Neuro-Symbolic Program Synthesis",
                })
            )
        );
        await waitFor(() =>
            expect(
                screen.getAllByText("Composable Neuro-Symbolic Inference").length
            ).toBeGreaterThan(0)
        );
    });

    test("renders error state when recommendations fail", async () => {
        const onRequestSimilarPapers = jest
            .fn()
            .mockRejectedValue(new Error("rate limit"));
        renderList({ onRequestSimilarPapers });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getAllByText("See similar papers")[0]);

        await waitFor(() =>
            expect(screen.getAllByText(/Failed to load recommendations/i).length).toBeGreaterThan(
                0
            )
        );
    });

    test("opens split paper reader modal and updates note text", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        const onResolvePaperMetadata = jest.fn().mockResolvedValue({
            url: "https://arxiv.org/abs/1706.03762",
        });
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "Seed note" })),
            onUpdatePaperAnnotation,
            onResolvePaperMetadata,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Seed note" }));

        expect(
            screen.getByRole("dialog", { name: "Paper reader and notes" })
        ).toBeTruthy();
        const richEditor = screen.getByRole("textbox", { name: "Paper notes editor" });
        richEditor.innerHTML = "<p>Expanded note text</p>";
        fireEvent.input(richEditor);

        expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
            "Neuro-Symbolic Program Synthesis",
            { notesMarkdown: "Expanded note text" }
        );
    });

    test("stores existing paper url on note-open when annotation url is missing", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            onUpdatePaperAnnotation,
            getPaperAnnotation: jest.fn(() => ({
                paperTitle: "Neuro-Symbolic Program Synthesis",
                notesMarkdown: "",
                sourceUrl: "",
            })),
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
                "Neuro-Symbolic Program Synthesis",
                { sourceUrl: "https://example.org/paper/neuro-symbolic" }
            )
        );
    });

    test("supports tab indentation and enter continuation for bullet notes", () => {
        window.localStorage.setItem("nesy_paper_notes_rich_editor_enabled", "0");
        const onUpdatePaperAnnotation = jest.fn();
        const onResolvePaperMetadata = jest.fn().mockResolvedValue({
            url: "https://arxiv.org/abs/1706.03762",
        });
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "- item" })),
            onUpdatePaperAnnotation,
            onResolvePaperMetadata,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "- item" }));
        const modalTextarea = screen.getByPlaceholderText(
            "Capture paper-specific insights. Use Tab/Shift+Tab for nested bullets."
        );

        modalTextarea.selectionStart = 0;
        modalTextarea.selectionEnd = "- item".length;
        fireEvent.keyDown(modalTextarea, { key: "Tab", code: "Tab" });
        expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
            "Neuro-Symbolic Program Synthesis",
            { notesMarkdown: "  - item" }
        );

        modalTextarea.selectionStart = "- item".length;
        modalTextarea.selectionEnd = "- item".length;
        fireEvent.keyDown(modalTextarea, { key: "Enter", code: "Enter" });
        expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
            "Neuro-Symbolic Program Synthesis",
            { notesMarkdown: "- item\n- " }
        );
    });

    test("rich editor invokes bold shortcut command", () => {
        const onUpdatePaperAnnotation = jest.fn();
        const originalExecCommand = document.execCommand;
        const execCommandSpy = jest.fn(() => true);
        document.execCommand = execCommandSpy;
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "Seed note" })),
            onUpdatePaperAnnotation,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Seed note" }));
        const richEditor = screen.getByRole("textbox", { name: "Paper notes editor" });

        fireEvent.keyDown(richEditor, { key: "b", ctrlKey: true });
        expect(execCommandSpy).toHaveBeenCalledWith("bold", false);
        document.execCommand = originalExecCommand;
    });

    test("rich editor turns dash-space into bullet list", () => {
        const onUpdatePaperAnnotation = jest.fn();
        const originalExecCommand = document.execCommand;
        const execCommandSpy = jest.fn(() => true);
        document.execCommand = execCommandSpy;
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "Seed note" })),
            onUpdatePaperAnnotation,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Seed note" }));
        const richEditor = screen.getByRole("textbox", { name: "Paper notes editor" });
        richEditor.innerHTML = "<p>-</p>";
        const textNode = richEditor.querySelector("p").firstChild;
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(textNode, textNode.textContent.length);
        range.collapse(true);
        selection.removeAllRanges();
        selection.addRange(range);

        fireEvent.keyDown(richEditor, { key: " ", code: "Space" });
        expect(execCommandSpy).toHaveBeenCalledWith("insertUnorderedList", false);
        document.execCommand = originalExecCommand;
    });

    test("uses desktop webview in desktop runtime with browser support", async () => {
        renderList({
            desktopConfig: { isDesktop: true, supportsInAppBrowser: true },
            requestedReaderItem: {
                title: "Remote paper",
                annotationKey: "Remote paper",
                url: "https://example.org/paper",
                status: "reading",
            },
            requestedReaderNonce: 1,
        });

        await waitFor(() =>
            expect(
                screen.getByTestId("desktop-paper-webview").textContent
            ).toContain("https://example.org/paper")
        );
    });

    test("to-read reader shows Mark as done and Close; mark ingests via callback", async () => {
        const onMarkReadingItemDone = jest
            .fn()
            .mockResolvedValue({ paper_title: "Neuro-Symbolic Program Synthesis" });
        renderList({
            requestedReaderItem: {
                readingItemId: "read-item-1",
                title: "Neuro-Symbolic Program Synthesis",
                annotationKey: "Neuro-Symbolic Program Synthesis",
                url: "https://arxiv.org/abs/1111.00001",
                authors: ["A. Author"],
                publication_date: "2024",
                venue: "ICLR",
                status: "inbox",
            },
            requestedReaderNonce: 1,
            readingItems: [
                {
                    id: "read-item-1",
                    sourceType: "url",
                    status: "inbox",
                    title: "Neuro-Symbolic Program Synthesis",
                    url: "https://arxiv.org/abs/1111.00001",
                    authors: ["A. Author"],
                    year: 2024,
                    venue: "ICLR",
                    topicHints: [],
                    linkedPaperTitle: null,
                    linkedThemeId: null,
                    semanticScholarPaperId: null,
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
            ],
            onMarkReadingItemDone,
        });

        await waitFor(() =>
            expect(screen.getByRole("dialog", { name: "Paper reader and notes" })).toBeTruthy()
        );
        expect(screen.getByRole("button", { name: "Mark as done" })).toBeTruthy();
        expect(screen.getByRole("button", { name: "Close" })).toBeTruthy();
        expect(screen.queryByRole("button", { name: "Done" })).toBeNull();

        fireEvent.click(screen.getByRole("button", { name: "Mark as done" }));

        await waitFor(() => expect(onMarkReadingItemDone).toHaveBeenCalledTimes(1));
        expect(onMarkReadingItemDone.mock.calls[0][0].id).toBe("read-item-1");
    });

    test("theme queue match by saved URL shows Mark as done when titles differ", async () => {
        const onMarkReadingItemDone = jest.fn().mockResolvedValue({});
        const sharedUrl = "https://arxiv.org/abs/2501.09999";
        renderList({
            selectedThemeId: "theme-neuro",
            papers: [
                {
                    title: "Graph Title After Ingest",
                    authors: ["A"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["T"],
                    url: "",
                },
            ],
            readingItems: [
                {
                    id: "theme-queue-1",
                    sourceType: "url",
                    status: "inbox",
                    linkedThemeId: "theme-neuro",
                    linkedPaperTitle: null,
                    title: "Different Scholar Title",
                    url: sharedUrl,
                    authors: [],
                    year: 2024,
                    venue: null,
                    topicHints: [],
                    semanticScholarPaperId: null,
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
            ],
            getPaperAnnotation: jest.fn(() => ({
                notesMarkdown: "",
                sourceUrl: sharedUrl,
            })),
            onMarkReadingItemDone,
            onResolvePaperMetadata: jest.fn(),
        });

        fireEvent.click(screen.getByText("Graph Title After Ingest"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(screen.getByRole("button", { name: "Mark as done" })).toBeTruthy()
        );
        fireEvent.click(screen.getByRole("button", { name: "Mark as done" }));
        await waitFor(() =>
            expect(onMarkReadingItemDone).toHaveBeenCalledWith(
                expect.objectContaining({ id: "theme-queue-1" })
            )
        );
    });

    test("migrates alias note to canonical title key for same paper", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            papers: [
                {
                    title: "Mesolimbic dopamine release conveys causal associations",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["Reward"],
                    url: "https://www.semanticscholar.org/paper/e0e4a9b215cade6d12c0c5579f996f3a4373c127",
                    semanticScholarPaperId: "e0e4a9b215cade6d12c0c5579f996f3a4373c127",
                },
            ],
            requestedPaperTitle: "Mesolimbic dopamine release conveys causal associations",
            requestedPaperNonce: 1,
            requestedReaderItem: {
                readingItemId: "queue-1",
                semanticScholarPaperId: "e0e4a9b215cade6d12c0c5579f996f3a4373c127",
                title: "science.abq6740",
                annotationKey: "science.abq6740",
                url: "https://doi.org/10.1126/science.abq6740",
                status: "inbox",
            },
            requestedReaderNonce: 1,
            getPaperAnnotation: jest.fn((key) => {
                if (key === "Mesolimbic dopamine release conveys causal associations") {
                    return {
                        paperTitle: key,
                        notesMarkdown: "",
                        sourceUrl:
                            "https://www.semanticscholar.org/paper/e0e4a9b215cade6d12c0c5579f996f3a4373c127",
                    };
                }
                if (key === "science.abq6740") {
                    return {
                        paperTitle: key,
                        notesMarkdown: "- recovered alias note",
                        sourceUrl: "https://doi.org/10.1126/science.abq6740",
                    };
                }
                return null;
            }),
            onUpdatePaperAnnotation,
        });

        await waitFor(() =>
            expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
                "Mesolimbic dopamine release conveys causal associations",
                expect.objectContaining({
                    notesMarkdown: "- recovered alias note",
                })
            )
        );
    });

    test("auto-resolves missing reader url when opening note card", async () => {
        const onResolvePaperMetadata = jest.fn().mockResolvedValue({
            title: "Neuro-Symbolic Program Synthesis",
            url: "https://arxiv.org/abs/2401.12345",
            authors: ["A. Author"],
            year: 2024,
            venue: "ICLR",
        });
        renderList({
            papers: [
                {
                    title: "Neuro-Symbolic Program Synthesis",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Combines neural and symbolic systems.",
                    topics: ["Neurosymbolic AI"],
                    url: "",
                },
            ],
            onResolvePaperMetadata,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(onResolvePaperMetadata).toHaveBeenCalledWith(
                expect.objectContaining({
                    title: "Neuro-Symbolic Program Synthesis",
                    authors: ["A. Author"],
                    year: 2024,
                })
            )
        );
        await waitFor(() =>
            expect(
                screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis")
            ).toBeTruthy()
        );
        expect(
            screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis").getAttribute(
                "src"
            )
        ).toBe("https://arxiv.org/abs/2401.12345");
    });

    test("shows manual link input when auto-resolve fails", async () => {
        renderList({
            papers: [
                {
                    title: "Neuro-Symbolic Program Synthesis",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Combines neural and symbolic systems.",
                    topics: ["Neurosymbolic AI"],
                    url: "",
                },
            ],
            onResolvePaperMetadata: jest
                .fn()
                .mockRejectedValue(new Error("Unable to resolve")),
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(screen.getByText("Unable to resolve")).toBeTruthy()
        );
        expect(
            screen.getByPlaceholderText("https://arxiv.org/abs/...")
        ).toBeTruthy();
    });

    test("manual link save loads reader and persists for next open", async () => {
        const onResolvePaperMetadata = jest
            .fn()
            .mockRejectedValue(new Error("Unable to resolve"));
        renderListWithAnnotationState({ onResolvePaperMetadata });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(
                screen.getByPlaceholderText("https://arxiv.org/abs/...")
            ).toBeTruthy()
        );
        fireEvent.change(screen.getByPlaceholderText("https://arxiv.org/abs/..."), {
            target: { value: "https://arxiv.org/abs/2501.00001" },
        });
        fireEvent.click(screen.getByRole("button", { name: "Save link" }));

        await waitFor(() =>
            expect(
                screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis")
            ).toBeTruthy()
        );
        expect(
            screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis").getAttribute(
                "src"
            )
        ).toBe("https://arxiv.org/abs/2501.00001");

        fireEvent.click(screen.getByRole("button", { name: "Done" }));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(
                screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis")
            ).toBeTruthy()
        );
        expect(
            screen.getByTitle("Paper content: Neuro-Symbolic Program Synthesis").getAttribute(
                "src"
            )
        ).toBe("https://arxiv.org/abs/2501.00001");
        expect(onResolvePaperMetadata).toHaveBeenCalledTimes(1);
    });

    test("reader annotation key prefers selected paper title over transient external key", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            papers: [
                {
                    title: "Canonical Graph Paper",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["T"],
                    url: "https://graph.example/paper",
                    semanticScholarPaperId: "paper-123",
                },
            ],
            requestedPaperTitle: "Canonical Graph Paper",
            requestedPaperNonce: 1,
            requestedReaderItem: {
                readingItemId: "read-1",
                title: "Publisher Redirect",
                annotationKey: "https://publisher.example/download?token=abc",
                url: "https://publisher.example/pdf/123",
                semanticScholarPaperId: "paper-123",
                status: "reading",
            },
            requestedReaderNonce: 1,
            readingItems: [
                {
                    id: "read-1",
                    sourceType: "url",
                    status: "reading",
                    title: "Publisher Redirect",
                    linkedPaperTitle: "Canonical Graph Paper",
                    linkedThemeId: null,
                    url: "https://publisher.example/pdf/123",
                    authors: [],
                    year: 2024,
                    venue: null,
                    topicHints: [],
                    semanticScholarPaperId: "paper-123",
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
            ],
            getPaperAnnotation: jest.fn((key) => ({
                paperTitle: key,
                notesMarkdown: "",
                sourceUrl: "",
            })),
            onUpdatePaperAnnotation,
        });

        await waitFor(() =>
            expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
                "Canonical Graph Paper",
                expect.objectContaining({ sourceUrl: "https://publisher.example/pdf/123" })
            )
        );
    });

    test("mark as done blocks ambiguous pending matches instead of picking first item", async () => {
        const onMarkReadingItemDone = jest.fn().mockResolvedValue({});
        renderList({
            papers: [
                {
                    title: "Canonical Graph Paper",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["T"],
                    url: "https://graph.example/paper",
                },
            ],
            selectedThemeId: "theme-1",
            getPaperAnnotation: jest.fn(() => ({
                notesMarkdown: "",
                sourceUrl: "https://resolved.example/paper",
            })),
            readingItems: [
                {
                    id: "candidate-1",
                    sourceType: "url",
                    status: "inbox",
                    title: "Candidate A",
                    linkedPaperTitle: null,
                    linkedThemeId: "theme-1",
                    url: "https://resolved.example/paper",
                    authors: [],
                    year: 2024,
                    venue: null,
                    topicHints: [],
                    semanticScholarPaperId: null,
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
                {
                    id: "candidate-2",
                    sourceType: "url",
                    status: "inbox",
                    title: "Candidate B",
                    linkedPaperTitle: null,
                    linkedThemeId: "theme-1",
                    url: "https://resolved.example/paper",
                    authors: [],
                    year: 2024,
                    venue: null,
                    topicHints: [],
                    semanticScholarPaperId: null,
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
            ],
            onResolvePaperMetadata: jest.fn(),
            onMarkReadingItemDone,
        });

        fireEvent.click(screen.getByText("Canonical Graph Paper"));
        fireEvent.click(
            screen.getByRole("button", {
                name: /No note yet\. Click to open the split paper reader popup\./i,
            })
        );

        await waitFor(() =>
            expect(screen.queryByRole("button", { name: "Mark as done" })).toBeNull()
        );
        expect(onMarkReadingItemDone).not.toHaveBeenCalled();
    });

    test("reader mark-as-done closes modal and drives shared ingesting tracker callbacks", async () => {
        const onMarkReadingItemDone = jest.fn().mockResolvedValue({ paper_title: "Paper One" });
        const onTrackIngestingItem = jest.fn();
        const onUntrackIngestingItem = jest.fn();
        renderList({
            requestedReaderItem: {
                readingItemId: "read-item-1",
                title: "Paper One",
                annotationKey: "Paper One",
                url: "https://arxiv.org/abs/1111.00001",
                status: "inbox",
            },
            requestedReaderNonce: 1,
            readingItems: [
                {
                    id: "read-item-1",
                    sourceType: "url",
                    status: "inbox",
                    title: "Paper One",
                    linkedPaperTitle: null,
                    linkedThemeId: null,
                    url: "https://arxiv.org/abs/1111.00001",
                    authors: [],
                    year: 2024,
                    venue: null,
                    topicHints: [],
                    semanticScholarPaperId: null,
                    quickNote: "",
                    createdAt: "2024-01-01T00:00:00Z",
                    updatedAt: "2024-01-01T00:00:00Z",
                },
            ],
            onMarkReadingItemDone,
            onTrackIngestingItem,
            onUntrackIngestingItem,
        });

        await waitFor(() =>
            expect(screen.getByRole("dialog", { name: "Paper reader and notes" })).toBeTruthy()
        );

        fireEvent.click(screen.getByRole("button", { name: "Mark as done" }));

        await waitFor(() => expect(onMarkReadingItemDone).toHaveBeenCalledTimes(1));
        expect(onTrackIngestingItem).toHaveBeenCalledWith(
            expect.objectContaining({ id: "read-item-1" })
        );
        expect(onUntrackIngestingItem).toHaveBeenCalledWith("read-item-1");
        await waitFor(() =>
            expect(screen.queryByRole("dialog", { name: "Paper reader and notes" })).toBeNull()
        );
    });

    test("external reader for different semantic scholar id does not use currently selected paper key", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            papers: [
                {
                    title: "Selected Paper",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["T"],
                    url: "https://www.semanticscholar.org/paper/Selected/abcd1234",
                    semanticScholarPaperId: "abcd1234",
                },
            ],
            requestedPaperTitle: "Selected Paper",
            requestedPaperNonce: 1,
            requestedReaderItem: {
                readingItemId: null,
                title: "Other paper title",
                annotationKey: "https://www.semanticscholar.org/paper/Other/efgh5678",
                semanticScholarPaperId: "efgh5678",
                url: "https://www.semanticscholar.org/paper/Other/efgh5678",
                status: "reading",
            },
            requestedReaderNonce: 1,
            getPaperAnnotation: jest.fn((key) => ({
                paperTitle: key,
                notesMarkdown: "",
                sourceUrl: "",
            })),
            onUpdatePaperAnnotation,
        });

        await waitFor(() =>
            expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
                "efgh5678",
                expect.objectContaining({
                    sourceUrl: "https://www.semanticscholar.org/paper/Other/efgh5678",
                })
            )
        );
        expect(onUpdatePaperAnnotation).not.toHaveBeenCalledWith(
            "Selected Paper",
            expect.anything()
        );
    });

    test("does not migrate alias notes into selected paper when external reader is a different paper", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            papers: [
                {
                    title: "Selected Paper",
                    authors: ["A. Author"],
                    publication_date: "2024",
                    abstract: "Abstract",
                    topics: ["T"],
                    url: "https://www.semanticscholar.org/paper/Selected/abcd1234",
                    semanticScholarPaperId: "abcd1234",
                },
            ],
            requestedPaperTitle: "Selected Paper",
            requestedPaperNonce: 1,
            requestedReaderItem: {
                readingItemId: null,
                title: "Other paper title",
                annotationKey: "https://www.semanticscholar.org/paper/Other/efgh5678",
                semanticScholarPaperId: "efgh5678",
                url: "https://www.semanticscholar.org/paper/Other/efgh5678",
                status: "reading",
            },
            requestedReaderNonce: 1,
            getPaperAnnotation: jest.fn((key) => {
                if (key === "Selected Paper") {
                    return { paperTitle: key, notesMarkdown: "", sourceUrl: "" };
                }
                if (key === "https://www.semanticscholar.org/paper/Other/efgh5678") {
                    return {
                        paperTitle: key,
                        notesMarkdown: "- other paper notes",
                        sourceUrl: "https://www.semanticscholar.org/paper/Other/efgh5678",
                    };
                }
                return { paperTitle: key, notesMarkdown: "", sourceUrl: "" };
            }),
            onUpdatePaperAnnotation,
        });

        await waitFor(() =>
            expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
                "efgh5678",
                expect.objectContaining({
                    sourceUrl: "https://www.semanticscholar.org/paper/Other/efgh5678",
                })
            )
        );
        expect(onUpdatePaperAnnotation).not.toHaveBeenCalledWith(
            "Selected Paper",
            expect.objectContaining({
                notesMarkdown: "- other paper notes",
            })
        );
    });
});
