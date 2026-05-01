import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import PaperWorkbenchList from "./PaperWorkbenchList";

function renderList(overrides = {}) {
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

describe("PaperWorkbenchList recommendations", () => {
    test("fetches similar papers for selected paper", async () => {
        const onRequestSimilarPapers = jest.fn().mockResolvedValue([
            { paperId: "rec-1", title: "Composable Neuro-Symbolic Inference" },
        ]);
        renderList({ onRequestSimilarPapers });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByText("See similar papers"));

        await waitFor(() =>
            expect(onRequestSimilarPapers).toHaveBeenCalledWith(
                expect.objectContaining({
                    title: "Neuro-Symbolic Program Synthesis",
                })
            )
        );
        await waitFor(() =>
            expect(
                screen.getByText("Composable Neuro-Symbolic Inference")
            ).toBeTruthy()
        );
    });

    test("renders error state when recommendations fail", async () => {
        const onRequestSimilarPapers = jest
            .fn()
            .mockRejectedValue(new Error("rate limit"));
        renderList({ onRequestSimilarPapers });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByText("See similar papers"));

        await waitFor(() =>
            expect(screen.getByText(/Failed to load recommendations/i)).toBeTruthy()
        );
    });

    test("opens full note editor modal and updates note text", async () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "Seed note" })),
            onUpdatePaperAnnotation,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Seed note" }));

        expect(screen.getByRole("dialog")).toBeTruthy();
        const modalTextarea = screen.getByPlaceholderText(
            "Capture paper-specific insights and how they connect to topics."
        );
        fireEvent.change(modalTextarea, { target: { value: "Expanded note text" } });

        expect(onUpdatePaperAnnotation).toHaveBeenCalledWith(
            "Neuro-Symbolic Program Synthesis",
            { notesMarkdown: "Expanded note text" }
        );
    });
});
