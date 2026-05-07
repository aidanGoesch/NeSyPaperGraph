import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ThemeNotebook from "./ThemeNotebook";

function renderNotebook(overrides = {}) {
    const props = {
        themeNotes: [
            {
                id: "theme-1",
                themeTitle: "Neurosymbolic Methods",
                linkedPaperTitles: ["Paper A"],
                sections: { notes: "theme notes", toRead: "" },
            },
        ],
        selectedThemeId: "theme-1",
        themeQueueItems: [],
        onSelectTheme: jest.fn(),
        onUpsertTheme: jest.fn((draft) => ({ ...draft, id: draft.id || "theme-1" })),
        onReorderReadingItem: jest.fn(),
        onSelectThemePaper: jest.fn(),
        onOpenThemeQueueItem: jest.fn(),
        onRequestThemeRecommendations: jest.fn().mockResolvedValue([]),
        ...overrides,
    };
    render(<ThemeNotebook {...props} />);
    return props;
}

describe("ThemeNotebook recommendations", () => {
    test("requests recommendations for selected theme", async () => {
        const onRequestThemeRecommendations = jest.fn().mockResolvedValue([
            { paperId: "theme-rec-1", title: "Logic-Enhanced Transformers" },
        ]);
        renderNotebook({ onRequestThemeRecommendations });

        fireEvent.click(screen.getByText("Recommend papers for theme"));

        await waitFor(() =>
            expect(onRequestThemeRecommendations).toHaveBeenCalledWith("theme-1")
        );
        await waitFor(() =>
            expect(screen.getByText("Logic-Enhanced Transformers")).toBeTruthy()
        );
    });

    test("shows recommendation error", async () => {
        const onRequestThemeRecommendations = jest
            .fn()
            .mockRejectedValue(new Error("upstream unavailable"));
        renderNotebook({ onRequestThemeRecommendations });

        fireEvent.click(screen.getByText("Recommend papers for theme"));

        await waitFor(() =>
            expect(screen.getByText(/Theme recommendations failed/i)).toBeTruthy()
        );
    });

    test("queue title click notifies open handler", () => {
        const onOpenThemeQueueItem = jest.fn();
        renderNotebook({
            themeQueueItems: [
                {
                    id: "queue-1",
                    sourceType: "url",
                    status: "inbox",
                    linkedThemeId: "theme-1",
                    linkedPaperTitle: null,
                    title: "Queued Paper Alpha",
                    url: "https://example.org/p",
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
            onOpenThemeQueueItem,
        });

        fireEvent.click(screen.getByRole("button", { name: "Queued Paper Alpha" }));
        expect(onOpenThemeQueueItem).toHaveBeenCalledWith(
            expect.objectContaining({ id: "queue-1" })
        );
    });

    test("shows empty-state message when no recommendations are returned", async () => {
        renderNotebook({ onRequestThemeRecommendations: jest.fn().mockResolvedValue([]) });

        fireEvent.click(screen.getByText("Recommend papers for theme"));

        await waitFor(() =>
            expect(
                screen.getByText(/No recommendations found for this theme yet/i)
            ).toBeTruthy()
        );
    });
});
