import { resolvePaperAnnotationKey } from "./readerIdentity";

describe("readerIdentity", () => {
    test("does not fallback to URL key by default", () => {
        const key = resolvePaperAnnotationKey({
            paperTitle: "",
            semanticScholarPaperId: "",
            url: "https://example.org/paper",
            fallbackTitle: "",
            fallbackKey: "fallback-key",
        });
        expect(key).toBe("fallback-key");
    });

    test("uses URL fallback only when explicitly enabled", () => {
        const key = resolvePaperAnnotationKey({
            paperTitle: "",
            semanticScholarPaperId: "",
            url: "https://example.org/paper#section",
            fallbackTitle: "",
            fallbackKey: "",
            allowUrlFallback: true,
        });
        expect(key).toBe("https://example.org/paper");
    });

    test("prefers semantic scholar id over URL fallback", () => {
        const key = resolvePaperAnnotationKey({
            paperTitle: "",
            semanticScholarPaperId: "paper-123",
            url: "https://example.org/paper",
            fallbackTitle: "",
            fallbackKey: "fallback-key",
            allowUrlFallback: true,
        });
        expect(key).toBe("paper-123");
    });
});
