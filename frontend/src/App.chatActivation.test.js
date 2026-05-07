import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

jest.mock("react-markdown", () => (props) => <div>{props.children}</div>);
jest.mock("marked", () => ({
    marked: (text) => text,
}));
jest.mock("mermaid", () => ({
    initialize: jest.fn(),
    render: jest.fn(async () => ({ svg: "<svg></svg>" })),
}));

jest.mock("./components/topic-workspace/TopicWorkspace", () => () => (
    <div>TopicWorkspace</div>
));

jest.mock("./state/workspaceStore", () => ({
    useWorkspaceStore: () => ({
        state: { readingItems: [], themeNotes: [] },
        actions: {
            addReadingItem: jest.fn(),
        },
        selectors: {},
        syncWarning: null,
    }),
}));

const mockActivationFrameSpy = jest.fn();

jest.mock("./GraphVisualization", () => {
    const ReactModule = require("react");
    return ReactModule.forwardRef((props, ref) => {
        mockActivationFrameSpy(props.activationFrame);
        ReactModule.useImperativeHandle(ref, () => ({
            focusOnPaper: jest.fn(),
            focusOnTopic: jest.fn(),
        }));
        return (
            <div data-testid="graph-visualization">
                Graph round {props.activationFrame?.roundIndex || 0}
            </div>
        );
    });
});

const App = require("./App").default;

describe("App chat-first activation mode", () => {
    beforeEach(() => {
        localStorage.setItem("nesy_access_key", "test-key");
        window.HTMLElement.prototype.scrollIntoView = jest.fn();
        global.EventSource = jest.fn(() => ({
            close: jest.fn(),
            addEventListener: jest.fn(),
            removeEventListener: jest.fn(),
        }));
        global.fetch = jest.fn(async (url, options) => {
            const endpoint = String(url);
            if (endpoint.includes("/api/runtime/diagnostics")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ status: "ok" }),
                };
            }
            if (endpoint.includes("/api/agent/architecture")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ diagram: "graph TD;A-->B;" }),
                };
            }
            if (endpoint.includes("/api/graph/load")) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        papers: [
                            {
                                title: "Paper A",
                                authors: ["Ada Lovelace"],
                                publication_date: "2024",
                                topics: ["Topic X"],
                                abstract: "Summary",
                            },
                        ],
                        topics: ["Topic X"],
                        edges: [{ source: "Paper A", target: "Topic X", type: "topic" }],
                    }),
                };
            }
            if (endpoint.includes("/api/search")) {
                const requestBody = JSON.parse(options?.body || "{}");
                expect(requestBody.activation_mode).toBe(true);
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        status: "success",
                        answer: "Here is an answer grounded in memory activation.",
                        final_answer: "Here is an answer grounded in memory activation.",
                        answer_structured: {
                            segments: [
                                { text: "A concise answer says ", claim_id: null },
                                { text: "Topic X is central", claim_id: "claim_topic_x" },
                                { text: " for this graph.", claim_id: null },
                            ],
                            claims: [
                                {
                                    id: "claim_topic_x",
                                    text: "Topic X is central",
                                    citations: [
                                        {
                                            paper_title: "Paper A",
                                            excerpt: "Topic X appears in the paper's main framing.",
                                        },
                                    ],
                                },
                            ],
                            warnings: [],
                        },
                        confidence: 0.74,
                        needs_more_context: false,
                        rounds: [
                            {
                                round_index: 1,
                                seed_nodes: [{ node_id: "Paper A", score: 0.92 }],
                                activated_nodes: [{ node_id: "Paper A", score: 1.0 }],
                                step_trace: [
                                    { step: 0, node_id: "Paper A", score_after_step: 1.0 },
                                    { step: 1, node_id: "Topic X", score_after_step: 0.64 },
                                ],
                            },
                        ],
                        sources_used: ["Paper A"],
                    }),
                };
            }
            return {
                ok: true,
                status: 200,
                json: async () => ({}),
            };
        });
    });

    afterEach(() => {
        jest.clearAllMocks();
        mockActivationFrameSpy.mockClear();
        localStorage.clear();
        delete global.EventSource;
    });

    test("renders graph and persistent chat panel together", async () => {
        render(<App />);
        await waitFor(() =>
            expect(screen.getByTestId("graph-visualization")).toBeTruthy()
        );
        expect(screen.getByPlaceholderText("Ask a follow-up question...")).toBeTruthy();
    });

    test("sends activation-mode search and surfaces confidence/round info", async () => {
        render(<App />);
        await waitFor(() =>
            expect(screen.getByTestId("graph-visualization")).toBeTruthy()
        );

        const input = screen.getByPlaceholderText("Ask a follow-up question...");
        fireEvent.change(input, { target: { value: "What did I read about Topic X?" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

        await waitFor(() =>
            expect(screen.getByText(/Confidence: 74%/)).toBeTruthy()
        );
        expect(screen.getByText(/Retrieval rounds: 1/)).toBeTruthy();
        await waitFor(() =>
            expect(screen.getByText(/Graph round 1/)).toBeTruthy()
        );
        expect(mockActivationFrameSpy).toHaveBeenCalledWith(
            expect.objectContaining({
                cameraPhase: "source_focus",
                traceNodeIds: expect.arrayContaining(["Paper A", "Topic X"]),
                focusNodeId: "Paper A",
            })
        );
        await waitFor(() =>
            expect(mockActivationFrameSpy).toHaveBeenCalledWith(
                expect.objectContaining({
                    cameraPhase: "zoomed_out_context",
                    traceNodeIds: expect.arrayContaining(["Paper A", "Topic X"]),
                })
            )
        );
    });

    test("renders interactive claims and opens inline evidence popover", async () => {
        render(<App />);
        await waitFor(() =>
            expect(screen.getByTestId("graph-visualization")).toBeTruthy()
        );

        const input = screen.getByPlaceholderText("Ask a follow-up question...");
        fireEvent.change(input, { target: { value: "What is Topic X?" } });
        fireEvent.keyDown(input, { key: "Enter", code: "Enter" });

        const claim = await waitFor(() => screen.getByText("Topic X is central"));
        fireEvent.click(claim);

        await waitFor(() =>
            expect(screen.getByText("Evidence")).toBeTruthy()
        );
        expect(
            screen.getByText(/Topic X appears in the paper's main framing\./)
        ).toBeTruthy();
        expect(screen.getAllByText("Paper A").length).toBeGreaterThan(0);
    });
});
