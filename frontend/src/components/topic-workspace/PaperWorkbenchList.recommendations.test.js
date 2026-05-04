import React from "react";
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
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "Seed note" })),
            onUpdatePaperAnnotation,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Open paper" }));

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

    test("supports tab indentation and enter continuation for bullet notes", () => {
        const onUpdatePaperAnnotation = jest.fn();
        renderList({
            getPaperAnnotation: jest.fn(() => ({ notesMarkdown: "- item" })),
            onUpdatePaperAnnotation,
        });

        fireEvent.click(screen.getByText("Neuro-Symbolic Program Synthesis"));
        fireEvent.click(screen.getByRole("button", { name: "Open paper" }));
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
});
