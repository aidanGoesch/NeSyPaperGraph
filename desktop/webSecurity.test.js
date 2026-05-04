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
    });

    const event = { prevented: false, preventDefault() { this.prevented = true; } };
    runtime.handleWillNavigate(event, "file:///private/etc/passwd", { kind: "webview" });
    assert.equal(event.prevented, true);
});
