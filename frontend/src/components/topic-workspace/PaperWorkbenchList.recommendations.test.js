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
        const modalTextarea = screen.getByPlaceholderText(
            "Capture paper-specific insights. Use Tab/Shift+Tab for nested bullets."
        );
        fireEvent.change(modalTextarea, { target: { value: "Expanded note text" } });

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
});
