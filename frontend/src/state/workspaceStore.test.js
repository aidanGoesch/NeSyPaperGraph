import React from "react";
import { act, render } from "@testing-library/react";
import { useWorkspaceStore } from "./workspaceStore";

let latestStore = null;

function StoreHarness() {
    latestStore = useWorkspaceStore({ isEnabled: false });
    return <div>workspace-store-harness</div>;
}

describe("workspaceStore reading list dedupe", () => {
    beforeEach(() => {
        latestStore = null;
        localStorage.clear();
    });

    test("does not add duplicate reading items for same semantic scholar paper", () => {
        render(<StoreHarness />);
        act(() => {
            latestStore.actions.addReadingItem({
                title: "Neural Program Repair for Code",
                semanticScholarPaperId: "S2-123",
                authors: ["Alice Chen"],
                year: 2023,
                status: "inbox",
            });
        });
        act(() => {
            latestStore.actions.addReadingItem({
                title: "Neural Program Repair for Code",
                semanticScholarPaperId: "S2-123",
                status: "inbox",
            });
        });
        expect(latestStore.state.readingItems).toHaveLength(1);
    });
});
