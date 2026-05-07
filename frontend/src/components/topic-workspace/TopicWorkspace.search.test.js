import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

jest.mock("./ClusterTree", () => (props) => (
    <div>
        <button type="button" onClick={() => props.onSelectCluster("cluster-1")}>
            Select Cluster
        </button>
    </div>
));

let lastPaperWorkbenchProps = null;
let lastThemeNotebookProps = null;
let lastToReadInboxProps = null;
jest.mock("./PaperWorkbenchList", () => (props) => {
    lastPaperWorkbenchProps = props;
    return <div>PaperWorkbenchList</div>;
});
jest.mock("./ThemeNotebook", () => (props) => {
    lastThemeNotebookProps = props;
    return <div>ThemeNotebook</div>;
});
jest.mock("./ToReadInbox", () => (props) => {
    lastToReadInboxProps = props;
    return <div>ToReadInbox</div>;
});
jest.mock("./ThemeAssignmentModal", () => () => <div>ThemeAssignmentModal</div>);

const TopicWorkspace = require("./TopicWorkspace").default;

function makeWorkspaceStore() {
    return {
        state: { readingItems: [], themeNotes: [] },
        actions: {
            upsertPaperAnnotation: jest.fn(),
            upsertThemeNote: jest.fn(),
            reorderReadingItem: jest.fn(),
            addReadingItem: jest.fn(),
            updateReadingItem: jest.fn(),
            removeReadingItem: jest.fn(),
            linkPaperToTheme: jest.fn(),
            setPaperThemeMembership: jest.fn(),
        },
        selectors: {
            getPaperAnnotation: jest.fn(() => null),
        },
    };
}

function makeGraphData() {
    return {
        topics: ["Neurosymbolic AI", "Program Repair"],
        papers: [
            {
                title: "Neurosymbolic Reasoning with Logic Programs",
                authors: ["Alice Chen", "David Bornstein"],
                publication_date: "2024",
                topics: ["Neurosymbolic AI"],
                abstract: "Combines neural and symbolic methods.",
            },
            {
                title: "Neural Program Repair for Code",
                authors: ["Alice Chen"],
                publication_date: "2023",
                topics: ["Program Repair"],
                abstract: "Program repair with neural models.",
            },
        ],
    };
}

function renderWorkspace(overrides = {}) {
    const props = {
        graphData: makeGraphData(),
        workspaceStore: makeWorkspaceStore(),
        onFocusPaper: jest.fn(),
        onSetGraphHighlight: jest.fn(),
        onResolveReadingUrl: jest.fn(),
        onIngestReadingItem: jest.fn(),
        apiBase: "http://localhost:8000",
        apiFetch: jest.fn(),
        ...overrides,
    };
    render(<TopicWorkspace {...props} />);
    return props;
}

describe("TopicWorkspace inferred search", () => {
    beforeEach(() => {
        jest.useFakeTimers();
        lastPaperWorkbenchProps = null;
        lastThemeNotebookProps = null;
        lastToReadInboxProps = null;
        delete window.desktopBridge;
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.clearAllMocks();
        delete window.desktopBridge;
    });

    test("renders search input and submits inferred query on Enter", async () => {
        const apiFetch = jest.fn().mockResolvedValue({
            ok: true,
            json: async () => ({ status: "success", results: [] }),
        });
        renderWorkspace({ apiFetch });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "chen and bornstein 2024" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });

        await waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(1));
        expect(apiFetch.mock.calls[0][0]).toContain("/api/topic-search");
    });

    test("shows loading state then renders result rows with metadata", async () => {
        const apiFetch = jest.fn().mockImplementation(
            () =>
                new Promise((resolve) =>
                    setTimeout(
                        () =>
                            resolve({
                                ok: true,
                                json: async () => ({
                                    status: "success",
                                    results: [
                                        {
                                            title: "Neurosymbolic Reasoning with Logic Programs",
                                            authors: ["Alice Chen", "David Bornstein"],
                                            publication_date: "2024",
                                            topics: ["Neurosymbolic AI", "Logic Programming"],
                                            summary: "A neurosymbolic paper.",
                                            score: 0.93,
                                            score_breakdown: {},
                                        },
                                    ],
                                }),
                            }),
                        50
                    )
                )
        );
        renderWorkspace({ apiFetch });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "neurosymbolic ai" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(300);
        });
        await waitFor(() =>
            expect(screen.getByText("Searching topic workspace...")).toBeTruthy()
        );
        act(() => {
            jest.advanceTimersByTime(60);
        });

        await waitFor(() =>
            expect(
                screen.getByText("Neurosymbolic Reasoning with Logic Programs")
            ).toBeTruthy()
        );
        expect(screen.getByText(/Alice Chen/)).toBeTruthy();
        expect(screen.getByText(/Neurosymbolic AI/)).toBeTruthy();
    });

    test("shows empty and error states", async () => {
        const apiFetch = jest
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ status: "success", results: [] }),
            })
            .mockResolvedValueOnce({
                ok: false,
                status: 500,
                json: async () => ({ detail: "server error" }),
            });
        renderWorkspace({ apiFetch });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "unknown query" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });
        await waitFor(() =>
            expect(screen.getByText("No matching papers found.")).toBeTruthy()
        );

        fireEvent.change(input, { target: { value: "force error" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });
        await waitFor(() =>
            expect(screen.getByText(/Topic search failed:/)).toBeTruthy()
        );
    });

    test("clicking a result opens paper in workspace panel", async () => {
        const onFocusPaper = jest.fn();
        const apiFetch = jest.fn().mockResolvedValue({
            ok: true,
            json: async () => ({
                status: "success",
                results: [
                    {
                        title: "Neural Program Repair for Code",
                        authors: ["Alice Chen"],
                        publication_date: "2023",
                        topics: ["Program Repair"],
                        summary: "Repair methods.",
                        score: 0.72,
                        score_breakdown: {},
                    },
                ],
            }),
        });
        renderWorkspace({ apiFetch, onFocusPaper });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "program repair chen" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });

        await waitFor(() =>
            expect(screen.getByText("Neural Program Repair for Code")).toBeTruthy()
        );
        fireEvent.click(screen.getByText("Open paper"));
        expect(onFocusPaper).not.toHaveBeenCalled();
        expect(lastPaperWorkbenchProps.requestedPaperTitle).toBe(
            "Neural Program Repair for Code"
        );
    });

    test("search result open uses normalized title lookup", async () => {
        const apiFetch = jest.fn().mockResolvedValue({
            ok: true,
            json: async () => ({
                status: "success",
                results: [
                    {
                        title: "  neural program repair for code  ",
                        authors: ["Alice Chen"],
                        publication_date: "2023",
                        topics: ["Program Repair"],
                        summary: "Repair methods.",
                        score: 0.72,
                        score_breakdown: {},
                    },
                ],
            }),
        });
        renderWorkspace({ apiFetch });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "program repair chen" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });

        await waitFor(() =>
            expect(screen.getByText(/neural program repair for code/i)).toBeTruthy()
        );
        fireEvent.click(screen.getByText("Open paper"));
        expect(lastPaperWorkbenchProps.requestedPaperTitle).toBe(
            "Neural Program Repair for Code"
        );
    });

    test("clicking a search result opens details popup and adds to reading list", async () => {
        const workspaceStore = makeWorkspaceStore();
        const apiFetch = jest.fn().mockResolvedValue({
            ok: true,
            json: async () => ({
                status: "success",
                results: [
                    {
                        title: "Neural Program Repair for Code",
                        authors: ["Alice Chen"],
                        publication_date: "2023",
                        topics: ["Program Repair"],
                        summary: "Repair methods.",
                        paperId: "S2-123",
                    },
                ],
            }),
        });
        renderWorkspace({ apiFetch, workspaceStore });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "program repair chen" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });

        await waitFor(() =>
            expect(screen.getByText("Neural Program Repair for Code")).toBeTruthy()
        );
        fireEvent.click(screen.getByText("Neural Program Repair for Code"));

        expect(screen.getByRole("dialog")).toBeTruthy();
        expect(screen.getByText("Repair methods.")).toBeTruthy();

        fireEvent.click(screen.getByText("Add to reading list"));
        expect(workspaceStore.actions.addReadingItem).toHaveBeenCalledWith(
            expect.objectContaining({
                title: "Neural Program Repair for Code",
                semanticScholarPaperId: "S2-123",
                status: "inbox",
            })
        );
    });

    test("query resets when cluster selection changes", async () => {
        const apiFetch = jest.fn().mockResolvedValue({
            ok: true,
            json: async () => ({ status: "success", results: [] }),
        });
        renderWorkspace({ apiFetch });

        const input = screen.getByPlaceholderText("Search papers, authors, or topics");
        fireEvent.change(input, { target: { value: "neurosymbolic ai" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });
        act(() => {
            jest.advanceTimersByTime(350);
        });
        await waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(1));

        fireEvent.click(screen.getByText("Select Cluster"));
        expect(screen.getByPlaceholderText("Search papers, authors, or topics").value).toBe("");
    });

    test("topic-level recommendations call backend and render results", async () => {
        const openSpy = jest.spyOn(window, "open").mockImplementation(() => null);
        const apiFetch = jest
            .fn()
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    status: "success",
                    results: [
                        {
                            paperId: "topic-rec-1",
                            title: "Neurosymbolic Agents with Planning",
                            year: 2025,
                        },
                    ],
                }),
            });
        renderWorkspace({
            apiFetch,
            workspaceStore: makeWorkspaceStore(),
        });

        fireEvent.click(
            screen.getByRole("button", { name: "Toggle topic action mode" })
        );
        fireEvent.change(
            screen.getByPlaceholderText(
                "Recommendation topic (e.g., causal reasoning)"
            ),
            {
                target: { value: "causal reasoning" },
            }
        );
        fireEvent.keyDown(
            screen.getByPlaceholderText(
                "Recommendation topic (e.g., causal reasoning)"
            ),
            { key: "Enter", code: "Enter" }
        );

        await waitFor(() =>
            expect(apiFetch).toHaveBeenCalledWith(
                expect.stringContaining("/api/recommendations/topic"),
                expect.objectContaining({
                    method: "POST",
                    body: expect.stringContaining("causal reasoning"),
                })
            )
        );
        await waitFor(() =>
            expect(
                screen.getByText("Neurosymbolic Agents with Planning")
            ).toBeTruthy()
        );
        fireEvent.click(screen.getByText("Neurosymbolic Agents with Planning"));
        expect(screen.getByRole("dialog")).toBeTruthy();
        fireEvent.click(screen.getAllByText("Open paper")[0]);
        expect(openSpy).toHaveBeenCalledWith(
            "https://www.semanticscholar.org/paper/topic-rec-1",
            "_blank",
            "noopener,noreferrer"
        );
        openSpy.mockRestore();
    });

    test("topic recommendations require an explicit topic", async () => {
        const apiFetch = jest.fn();
        renderWorkspace({
            apiFetch,
            workspaceStore: makeWorkspaceStore(),
        });

        fireEvent.click(
            screen.getByRole("button", { name: "Toggle topic action mode" })
        );
        fireEvent.keyDown(
            screen.getByPlaceholderText(
                "Recommendation topic (e.g., causal reasoning)"
            ),
            { key: "Enter", code: "Enter" }
        );
        expect(apiFetch).not.toHaveBeenCalled();
    });

    test("theme recommendation request falls back to workspace route on 404", async () => {
        const apiFetch = jest
            .fn()
            .mockResolvedValueOnce({
                ok: false,
                status: 404,
                json: async () => ({ detail: "Not Found" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    status: "success",
                    results: [{ paperId: "r-1", title: "Fallback rec" }],
                }),
            });
        const workspaceStore = makeWorkspaceStore();
        workspaceStore.state.themeNotes = [
            {
                id: "theme-1",
                themeTitle: "Theme One",
                linkedPaperTitles: [],
                sections: { notes: "", toRead: "" },
            },
        ];
        renderWorkspace({ apiFetch, workspaceStore });

        await lastThemeNotebookProps.onRequestThemeRecommendations("theme-1");

        expect(apiFetch.mock.calls[0][0]).toContain("/api/recommendations/theme");
        expect(apiFetch.mock.calls[1][0]).toContain(
            "/api/workspace/recommendations/theme"
        );
    });

    test("paper recommendation request falls back to workspace route on 404", async () => {
        const apiFetch = jest
            .fn()
            .mockResolvedValueOnce({
                ok: false,
                status: 404,
                json: async () => ({ detail: "Not Found" }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    status: "success",
                    results: [{ paperId: "pr-1", title: "Fallback similar paper" }],
                }),
            });
        renderWorkspace({ apiFetch, workspaceStore: makeWorkspaceStore() });

        await lastPaperWorkbenchProps.onRequestSimilarPapers({
            title: "Neural Program Repair for Code",
            authors: ["Alice Chen"],
            publication_date: "2023",
            abstract: "Program repair with neural models.",
        });

        expect(apiFetch.mock.calls[0][0]).toContain("/api/recommendations/paper");
        expect(apiFetch.mock.calls[1][0]).toContain(
            "/api/workspace/recommendations/paper"
        );
    });

    test("desktop webview open event routes url to reader popup", () => {
        let onOpen = null;
        window.desktopBridge = {
            onOpenInReaderUrl(handler) {
                onOpen = handler;
                return () => {
                    onOpen = null;
                };
            },
        };

        renderWorkspace({
            desktopConfig: {
                isDesktop: true,
                supportsInAppBrowser: true,
            },
        });

        act(() => {
            onOpen?.("https://arxiv.org/abs/1706.03762");
        });

        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                url: "https://arxiv.org/abs/1706.03762",
                status: "reading",
            })
        );
    });

    test("desktop open event reuses active identity for non-ssid publisher url hops", () => {
        let onOpen = null;
        window.desktopBridge = {
            onOpenInReaderUrl(handler) {
                onOpen = handler;
                return () => {
                    onOpen = null;
                };
            },
        };
        const workspaceStore = makeWorkspaceStore();
        workspaceStore.state.readingItems = [
            {
                id: "read-1",
                status: "inbox",
                title: "Paper Identity",
                linkedPaperTitle: "Paper Identity",
                url: "https://www.semanticscholar.org/paper/Paper-Identity/abcd1234",
                semanticScholarPaperId: "abcd1234",
                authors: ["A"],
                year: 2019,
                venue: "Conf",
            },
        ];
        renderWorkspace({
            workspaceStore,
            desktopConfig: {
                isDesktop: true,
                supportsInAppBrowser: true,
            },
        });

        act(() => {
            lastToReadInboxProps.onOpenReadingItem(workspaceStore.state.readingItems[0]);
        });
        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                readingItemId: "read-1",
                semanticScholarPaperId: "abcd1234",
            })
        );

        act(() => {
            onOpen?.("https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf");
        });
        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                readingItemId: "read-1",
                semanticScholarPaperId: "abcd1234",
                title: "Paper Identity",
                url: "https://www.wilmabainbridge.com/sharepapers/plm-2019.pdf",
            })
        );
    });

    test("desktop open event reuses identity when semantic scholar id matches", () => {
        let onOpen = null;
        window.desktopBridge = {
            onOpenInReaderUrl(handler) {
                onOpen = handler;
                return () => {
                    onOpen = null;
                };
            },
        };
        const workspaceStore = makeWorkspaceStore();
        workspaceStore.state.readingItems = [
            {
                id: "read-1",
                status: "reading",
                title: "Paper Identity",
                linkedPaperTitle: "Paper Identity",
                url: "https://www.semanticscholar.org/paper/Paper-Identity/abcd1234",
                semanticScholarPaperId: "abcd1234",
                authors: ["A"],
                year: 2019,
                venue: "Conf",
            },
        ];
        renderWorkspace({
            workspaceStore,
            desktopConfig: {
                isDesktop: true,
                supportsInAppBrowser: true,
            },
        });

        act(() => {
            lastToReadInboxProps.onOpenReadingItem(workspaceStore.state.readingItems[0]);
        });
        act(() => {
            onOpen?.("https://www.semanticscholar.org/paper/Paper-Identity-Variant/abcd1234");
        });
        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                readingItemId: "read-1",
                semanticScholarPaperId: "abcd1234",
                title: "Paper Identity",
            })
        );
    });

    test("desktop open event does not reuse active identity for different semantic scholar paper id", () => {
        let onOpen = null;
        window.desktopBridge = {
            onOpenInReaderUrl(handler) {
                onOpen = handler;
                return () => {
                    onOpen = null;
                };
            },
        };
        const workspaceStore = makeWorkspaceStore();
        workspaceStore.state.readingItems = [
            {
                id: "read-1",
                status: "reading",
                title: "Paper Identity",
                linkedPaperTitle: "Paper Identity",
                url: "https://www.semanticscholar.org/paper/Paper-Identity/abcd1234",
                semanticScholarPaperId: "abcd1234",
                authors: ["A"],
                year: 2019,
                venue: "Conf",
            },
        ];
        renderWorkspace({
            workspaceStore,
            desktopConfig: {
                isDesktop: true,
                supportsInAppBrowser: true,
            },
        });

        act(() => {
            lastToReadInboxProps.onOpenReadingItem(workspaceStore.state.readingItems[0]);
        });
        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                readingItemId: "read-1",
                semanticScholarPaperId: "abcd1234",
                title: "Paper Identity",
            })
        );

        act(() => {
            onOpen?.("https://www.semanticscholar.org/paper/Other-Paper/efgh5678");
        });
        expect(lastPaperWorkbenchProps.requestedReaderItem).toEqual(
            expect.objectContaining({
                readingItemId: null,
                semanticScholarPaperId: "efgh5678",
                url: "https://www.semanticscholar.org/paper/Other-Paper/efgh5678",
            })
        );
        expect(lastPaperWorkbenchProps.requestedReaderItem.title).not.toBe("Paper Identity");
    });

    test("markReadingItemDone migrates note content to ingested paper title when source key differs", async () => {
        const upsertPaperAnnotation = jest.fn();
        const removeReadingItem = jest.fn();
        const workspaceStore = {
            state: {
                readingItems: [
                    {
                        id: "read-1",
                        status: "inbox",
                        title: "Memorability alias",
                        linkedPaperTitle: null,
                        semanticScholarPaperId: "186336411",
                        url: "https://www.semanticscholar.org/paper/Foo/186336411",
                    },
                ],
                themeNotes: [],
                paperAnnotations: {},
            },
            actions: {
                upsertPaperAnnotation,
                upsertThemeNote: jest.fn(),
                reorderReadingItem: jest.fn(),
                addReadingItem: jest.fn(),
                updateReadingItem: jest.fn(),
                removeReadingItem,
                linkPaperToTheme: jest.fn(),
                setPaperThemeMembership: jest.fn(),
            },
            selectors: {
                getPaperAnnotation: jest.fn((key) => {
                    if (key === "ssid:186336411") {
                        return {
                            paperTitle: "ssid:186336411",
                            notesMarkdown: "- nested note\n  - detail",
                            sourceUrl:
                                "https://www.semanticscholar.org/paper/Foo/186336411",
                        };
                    }
                    if (key === "Memorability: How what we see influences what we remember") {
                        return {
                            paperTitle: key,
                            notesMarkdown: "",
                            sourceUrl: "",
                        };
                    }
                    return null;
                }),
            },
        };
        const onIngestReadingItem = jest.fn().mockResolvedValue({
            paper_title: "Memorability: How what we see influences what we remember",
        });

        renderWorkspace({ workspaceStore, onIngestReadingItem });
        await act(async () => {
            await lastPaperWorkbenchProps.onMarkReadingItemDone(workspaceStore.state.readingItems[0]);
        });

        expect(upsertPaperAnnotation).toHaveBeenCalledWith(
            "Memorability: How what we see influences what we remember",
            expect.objectContaining({
                notesMarkdown: "- nested note\n  - detail",
            })
        );
        expect(removeReadingItem).toHaveBeenCalledWith("read-1");
    });
});
