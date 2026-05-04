import React, { useEffect, useMemo, useRef, useState } from "react";

function formatReaderUrl(url) {
    if (!url) return "";
    try {
        const parsed = new URL(url);
        const preview = `${parsed.hostname}${parsed.pathname}`;
        return preview.length > 78 ? `${preview.slice(0, 78)}...` : preview;
    } catch {
        return url.length > 78 ? `${url.slice(0, 78)}...` : url;
    }
}

export default function DesktopPaperWebview({ url, title = "paper", className = "" }) {
    const webviewRef = useRef(null);
    const [isLoading, setIsLoading] = useState(Boolean(url));
    const [loadError, setLoadError] = useState("");
    const [canGoBack, setCanGoBack] = useState(false);
    const [canGoForward, setCanGoForward] = useState(false);
    const [currentUrl, setCurrentUrl] = useState(url || "");

    const mergedClassName = useMemo(
        () => `paper-reader-webview-container ${className}`.trim(),
        [className]
    );

    useEffect(() => {
        setIsLoading(Boolean(url));
        setLoadError("");
        setCurrentUrl(url || "");
        setCanGoBack(false);
        setCanGoForward(false);
    }, [url]);

    useEffect(() => {
        const webview = webviewRef.current;
        if (!webview) return undefined;

        const updateNavigationState = () => {
            try {
                setCanGoBack(Boolean(webview.canGoBack?.()));
                setCanGoForward(Boolean(webview.canGoForward?.()));
            } catch {
                setCanGoBack(false);
                setCanGoForward(false);
            }
            try {
                const nextUrl = webview.getURL?.();
                if (nextUrl) {
                    setCurrentUrl(nextUrl);
                }
            } catch {
                // no-op
            }
        };

        const onStart = () => {
            setIsLoading(true);
            setLoadError("");
            updateNavigationState();
        };
        const onStop = () => {
            setIsLoading(false);
            updateNavigationState();
        };
        const onFail = (event) => {
            setIsLoading(false);
            setLoadError(
                event?.errorDescription
                    ? `Unable to load page: ${event.errorDescription}`
                    : "Unable to load page in app. Use 'Open in browser'."
            );
            updateNavigationState();
        };
        const onNavigate = () => {
            updateNavigationState();
        };

        webview.addEventListener("did-start-loading", onStart);
        webview.addEventListener("did-stop-loading", onStop);
        webview.addEventListener("did-fail-load", onFail);
        webview.addEventListener("did-navigate", onNavigate);
        webview.addEventListener("did-navigate-in-page", onNavigate);

        updateNavigationState();

        return () => {
            webview.removeEventListener("did-start-loading", onStart);
            webview.removeEventListener("did-stop-loading", onStop);
            webview.removeEventListener("did-fail-load", onFail);
            webview.removeEventListener("did-navigate", onNavigate);
            webview.removeEventListener("did-navigate-in-page", onNavigate);
        };
    }, [url]);

    return (
        <div className={mergedClassName}>
            <div className="paper-reader-frame-toolbar">
                <div className="paper-reader-nav-controls">
                    <button
                        type="button"
                        className="open-link-button"
                        disabled={!canGoBack}
                        onClick={() => webviewRef.current?.goBack?.()}
                    >
                        Back
                    </button>
                    <button
                        type="button"
                        className="open-link-button"
                        disabled={!canGoForward}
                        onClick={() => webviewRef.current?.goForward?.()}
                    >
                        Forward
                    </button>
                    <button
                        type="button"
                        className="open-link-button"
                        onClick={() => webviewRef.current?.reload?.()}
                    >
                        Reload
                    </button>
                </div>
                <a
                    href={currentUrl || url}
                    target="_blank"
                    rel="noreferrer"
                    className="paper-reader-source-link"
                    title={currentUrl || url}
                >
                    {formatReaderUrl(currentUrl || url)}
                </a>
                <a
                    href={currentUrl || url}
                    target="_blank"
                    rel="noreferrer"
                    className="open-link-button"
                >
                    Open in browser
                </a>
            </div>
            {isLoading && <p className="theme-sync-hint">Loading paper page...</p>}
            {loadError && <p className="validation-error">{loadError}</p>}
            <webview
                ref={webviewRef}
                className="paper-reader-webview"
                src={url}
                allowpopups="true"
                title={`Paper content: ${title}`}
            />
        </div>
    );
}
