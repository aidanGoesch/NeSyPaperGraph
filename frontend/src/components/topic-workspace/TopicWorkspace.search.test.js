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
jest.mock("./PaperWorkbenchList", () => (props) => {
    lastPaperWorkbenchProps = props;
    return <div>PaperWorkbenchList</div>;
});
jest.mock("./ThemeNotebook", () => () => <div>ThemeNotebook</div>);
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
});
