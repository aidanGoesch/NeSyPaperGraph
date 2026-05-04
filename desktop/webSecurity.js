function isHttpLikeUrl(url) {
    try {
        const parsed = new URL(String(url || ""));
        return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
        return false;
    }
}

function isFileUrl(url) {
    try {
        const parsed = new URL(String(url || ""));
        return parsed.protocol === "file:";
    } catch {
        return false;
    }
}

function isSameOrigin(url, expectedOrigin) {
    if (!expectedOrigin) return false;
    try {
        return new URL(String(url || "")).origin === expectedOrigin;
    } catch {
        return false;
    }
}

function createSecurityPolicyRuntime({
    getMainWindowWebContents,
    getRendererEntry,
    openExternal,
    openInReader,
}) {
    const rendererEntry = getRendererEntry();
    const rendererOrigin =
        rendererEntry?.type === "url"
            ? new URL(rendererEntry.value).origin
            : null;

    function isRendererReferrer(referrerUrl) {
        if (!referrerUrl) return false;
        if (rendererEntry?.type === "file") {
            return isFileUrl(referrerUrl);
        }
        return isSameOrigin(referrerUrl, rendererOrigin);
    }

    function handleWindowOpen(details, sourceWebContents) {
        const url = details?.url || "";
        const referrerUrl = details?.referrer?.url || "";
        const mainWindowContents = getMainWindowWebContents();
        const isMainWindow = Boolean(
            mainWindowContents && sourceWebContents === mainWindowContents
        );
        if (isHttpLikeUrl(url)) {
            const rendererInitiated = isRendererReferrer(referrerUrl);
            if (isMainWindow && rendererInitiated) {
                openExternal(url);
            } else {
                openInReader(url);
            }
        }
        return { action: "deny" };
    }

    function handleWillNavigate(event, url, sourceWebContents) {
        const mainWindowContents = getMainWindowWebContents();
        const isMainWindow = Boolean(
            mainWindowContents && sourceWebContents === mainWindowContents
        );

        if (isMainWindow) {
            if (
                (rendererEntry?.type === "file" && isFileUrl(url)) ||
                (rendererEntry?.type === "url" && isSameOrigin(url, rendererOrigin))
            ) {
                return;
            }
            event.preventDefault();
            if (isHttpLikeUrl(url)) {
                openExternal(url);
            }
            return;
        }

        if (!isHttpLikeUrl(url)) {
            event.preventDefault();
        }
    }

    function handlePermissionRequest(_webContents, _permission, callback) {
        callback(false);
    }

    function handlePermissionCheck() {
        return false;
    }

    return {
        handleWindowOpen,
        handleWillNavigate,
        handlePermissionRequest,
        handlePermissionCheck,
    };
}

module.exports = {
    isHttpLikeUrl,
    createSecurityPolicyRuntime,
};
