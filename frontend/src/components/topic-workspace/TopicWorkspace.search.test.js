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
jest.mock("./PaperWorkbenchList", () => (props) => {
    lastPaperWorkbenchProps = props;
    return <div>PaperWorkbenchList</div>;
});
jest.mock("./ThemeNotebook", () => (props) => {
    lastThemeNotebookProps = props;
    return <div>ThemeNotebook</div>;
});
jest.mock("./ToReadInbox", () => () => <div>ToReadInbox</div>);
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
    });

    afterEach(() => {
        jest.runOnlyPendingTimers();
        jest.useRealTimers();
        jest.clearAllMocks();
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
});
