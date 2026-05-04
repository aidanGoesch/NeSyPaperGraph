import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import ToReadInbox from "./ToReadInbox";

function renderInbox(overrides = {}) {
    const props = {
        readingItems: [],
        topics: [],
        themeNotes: [],
        onAddReadingItem: jest.fn(),
        onUpdateReadingItem: jest.fn(),
        onRemoveReadingItem: jest.fn(),
        onReorderReadingItem: jest.fn(),
        onFocusPaper: jest.fn(),
        onResolveReadingUrl: jest.fn(),
        onMarkReadingItemDone: jest.fn(),
        onOpenReadingItem: jest.fn(),
        ...overrides,
    };
    render(<ToReadInbox {...props} />);
    return props;
}

describe("ToReadInbox open action", () => {
    test("open button launches popup reader callback for unread items", () => {
        const onOpenReadingItem = jest.fn();
        const readingItems = [
            {
                id: "r-1",
                title: "Paper One",
                url: "https://example.org/paper-one",
                status: "reading",
            },
        ];
        renderInbox({ readingItems, onOpenReadingItem });

        fireEvent.click(screen.getByRole("button", { name: "Open" }));
        expect(onOpenReadingItem).toHaveBeenCalledWith(
            expect.objectContaining({ id: "r-1" })
        );
    });

    test("done items do not render open button", () => {
        const readingItems = [
            {
                id: "r-2",
                title: "Paper Two",
                url: "https://example.org/paper-two",
                status: "done",
            },
        ];
        renderInbox({ readingItems });

        expect(screen.queryByRole("button", { name: "Open" })).toBeNull();
    });

    test("uses resolved metadata url when adding item", async () => {
        const onAddReadingItem = jest.fn();
        const onResolveReadingUrl = jest.fn().mockResolvedValue({
            url: "https://arxiv.org/abs/1706.03762",
            title: "Attention Is All You Need",
            semanticScholarPaperId: "s2-1",
            authors: ["Ashish Vaswani"],
            year: 2017,
            venue: "NeurIPS",
        });

        renderInbox({ onAddReadingItem, onResolveReadingUrl });

        fireEvent.change(
            screen.getByPlaceholderText("Paste paper URL"),
            { target: { value: "https://www.semanticscholar.org/paper/s2-1" } }
        );
        fireEvent.click(screen.getByRole("button", { name: "Add" }));

        await waitFor(() =>
            expect(onAddReadingItem).toHaveBeenCalledWith(
                expect.objectContaining({
                    url: "https://arxiv.org/abs/1706.03762",
                    semanticScholarPaperId: "s2-1",
                })
            )
        );
    });
});
