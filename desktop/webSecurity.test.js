const test = require("node:test");
const assert = require("node:assert/strict");
const {
    isHttpLikeUrl,
    createSecurityPolicyRuntime,
} = require("./webSecurity");

test("isHttpLikeUrl only allows http and https", () => {
    assert.equal(isHttpLikeUrl("https://example.org"), true);
    assert.equal(isHttpLikeUrl("http://example.org"), true);
    assert.equal(isHttpLikeUrl("file:///tmp/foo"), false);
    assert.equal(isHttpLikeUrl("javascript:alert(1)"), false);
    assert.equal(isHttpLikeUrl(""), false);
});

test("main window navigation blocks external loads and opens browser", () => {
    const opened = [];
    const mainWindowContents = {};
    const runtime = createSecurityPolicyRuntime({
        getMainWindowWebContents: () => mainWindowContents,
        getRendererEntry: () => ({
            type: "url",
            value: "http://localhost:3000",
        }),
        openExternal: (url) => opened.push(url),
        openInReader: () => {},
    });

    const event = { prevented: false, preventDefault() { this.prevented = true; } };
    runtime.handleWillNavigate(event, "https://example.org/paper", mainWindowContents);

    assert.equal(event.prevented, true);
    assert.deepEqual(opened, ["https://example.org/paper"]);
});

test("main window allows same-origin app navigation", () => {
    const mainWindowContents = {};
    const runtime = createSecurityPolicyRuntime({
        getMainWindowWebContents: () => mainWindowContents,
        getRendererEntry: () => ({
            type: "url",
            value: "http://localhost:3000",
        }),
        openExternal: () => {},
        openInReader: () => {},
    });

    const event = { prevented: false, preventDefault() { this.prevented = true; } };
    runtime.handleWillNavigate(event, "http://localhost:3000/workspace", mainWindowContents);
    assert.equal(event.prevented, false);
});

test("webview blocks non-http navigation", () => {
    const runtime = createSecurityPolicyRuntime({
        getMainWindowWebContents: () => ({}),
        getRendererEntry: () => ({
            type: "url",
            value: "http://localhost:3000",
        }),
        openExternal: () => {},
        openInReader: () => {},
    });

    const event = { prevented: false, preventDefault() { this.prevented = true; } };
    runtime.handleWillNavigate(event, "file:///private/etc/passwd", { kind: "webview" });
    assert.equal(event.prevented, true);
});

test("webview window-open routes http links to in-app reader", () => {
    const openedExternal = [];
    const openedInReader = [];
    const mainWindowContents = { kind: "main" };
    const webviewContents = { kind: "webview" };
    const runtime = createSecurityPolicyRuntime({
        getMainWindowWebContents: () => mainWindowContents,
        getRendererEntry: () => ({
            type: "url",
            value: "http://localhost:3000",
        }),
        openExternal: (url) => openedExternal.push(url),
        openInReader: (url) => openedInReader.push(url),
    });

    const result = runtime.handleWindowOpen(
        {
            url: "https://arxiv.org/abs/1706.03762",
            referrer: { url: "https://www.semanticscholar.org/paper/abc" },
        },
        webviewContents
    );

    assert.deepEqual(result, { action: "deny" });
    assert.deepEqual(openedExternal, []);
    assert.deepEqual(openedInReader, ["https://arxiv.org/abs/1706.03762"]);
});

test("main window renderer-initiated open stays external", () => {
    const openedExternal = [];
    const openedInReader = [];
    const mainWindowContents = { kind: "main" };
    const runtime = createSecurityPolicyRuntime({
        getMainWindowWebContents: () => mainWindowContents,
        getRendererEntry: () => ({
            type: "url",
            value: "http://localhost:3000",
        }),
        openExternal: (url) => openedExternal.push(url),
        openInReader: (url) => openedInReader.push(url),
    });

    const result = runtime.handleWindowOpen(
        {
            url: "https://example.org",
            referrer: { url: "http://localhost:3000/workspace" },
        },
        mainWindowContents
    );

    assert.deepEqual(result, { action: "deny" });
    assert.deepEqual(openedExternal, ["https://example.org"]);
    assert.deepEqual(openedInReader, []);
});
