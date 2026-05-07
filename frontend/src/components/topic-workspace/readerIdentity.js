export function normalizePaperTitle(value) {
    return String(value || "").trim().toLowerCase();
}

export function normalizeReaderLookupUrl(url) {
    if (!url || typeof url !== "string") return "";
    try {
        const parsed = new URL(url.trim());
        parsed.hash = "";
        const path = parsed.pathname.replace(/\/$/, "") || "/";
        return `${parsed.protocol}//${parsed.hostname.toLowerCase()}${path}${parsed.search}`;
    } catch {
        return String(url).trim().toLowerCase();
    }
}

export function normalizeSemanticScholarId(value) {
    return String(value || "").trim().toLowerCase();
}

export function extractSemanticScholarPaperIdFromUrl(url) {
    const source = String(url || "").trim();
    if (!source) return "";
    try {
        const parsed = new URL(source);
        const host = parsed.hostname.toLowerCase();
        if (!host.includes("semanticscholar.org")) return "";
        const segments = parsed.pathname.split("/").filter(Boolean);
        const paperIdx = segments.findIndex((segment) => segment.toLowerCase() === "paper");
        if (paperIdx < 0 || paperIdx + 2 >= segments.length) return "";
        return normalizeSemanticScholarId(segments[paperIdx + 2]);
    } catch {
        return "";
    }
}

export function resolvePaperAnnotationKey({
    paperTitle,
    semanticScholarPaperId,
    url,
    fallbackTitle,
    fallbackKey,
    allowUrlFallback = false,
}) {
    const title = String(paperTitle || "").trim();
    if (title) return title;

    const paperId = String(semanticScholarPaperId || "").trim();
    if (paperId) return paperId;

    if (allowUrlFallback) {
        const normalizedUrl = normalizeReaderLookupUrl(url || "");
        if (normalizedUrl) return normalizedUrl;
    }

    const fallbackText = String(fallbackTitle || "").trim();
    if (fallbackText) return fallbackText;
    return String(fallbackKey || "").trim() || null;
}
