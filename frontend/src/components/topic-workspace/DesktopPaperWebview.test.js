import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import DesktopPaperWebview from "./DesktopPaperWebview";

function attachWebviewMethods() {
    const webview = document.querySelector("webview");
    webview.canGoBack = jest.fn(() => true);
    webview.canGoForward = jest.fn(() => false);
    webview.goBack = jest.fn();
    webview.goForward = jest.fn();
    webview.reload = jest.fn();
    webview.getURL = jest.fn(() => "https://example.org/next");
    return webview;
}

describe("DesktopPaperWebview", () => {
    test("renders toolbar and webview", () => {
        render(
            <DesktopPaperWebview
                url="https://example.org/paper"
                title="Example paper"
            />
        );
        const webview = attachWebviewMethods();

        expect(webview).toBeTruthy();
        expect(screen.getByText("Open in browser")).toBeTruthy();
    });

    test("updates loading and navigation state from webview events", () => {
        render(
            <DesktopPaperWebview
                url="https://example.org/paper"
                title="Example paper"
            />
        );
        const webview = attachWebviewMethods();

        fireEvent(webview, new Event("did-stop-loading"));
        expect(screen.queryByText("Loading paper page...")).toBeNull();
        expect(screen.getByText("example.org/next")).toBeTruthy();
    });

    test("shows error text on failed load event", () => {
        render(
            <DesktopPaperWebview
                url="https://example.org/paper"
                title="Example paper"
            />
        );
        const webview = attachWebviewMethods();

        const failEvent = new Event("did-fail-load");
        failEvent.errorDescription = "ERR_BLOCKED_BY_RESPONSE";
        fireEvent(webview, failEvent);

        expect(screen.getByText(/Unable to load page:/)).toBeTruthy();
    });
});
