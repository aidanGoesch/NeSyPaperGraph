import {
    backupPaperNoteSnapshot,
    editableHtmlToMarkdown,
    handleListTabIndent,
    markdownRoundTrip,
    markdownToEditableHtml,
} from "./paperNotesRichText";

describe("paperNotesRichText safety and conversion", () => {
    beforeEach(() => {
        window.localStorage.removeItem("nesy_paper_notes_backup_v1");
    });

    test("backs up note snapshots without overwriting identical content", () => {
        backupPaperNoteSnapshot("paper-1", "- item");
        backupPaperNoteSnapshot("paper-1", "- item");
        backupPaperNoteSnapshot("paper-1", "- updated");

        const parsed = JSON.parse(window.localStorage.getItem("nesy_paper_notes_backup_v1"));
        expect(parsed["paper-1"].notesMarkdown).toBe("- updated");
    });

    test("round-trips representative markdown fixtures", () => {
        const fixtures = [
            "- parent\n    - child",
            "- item with wrap\n  continuation line",
            "Mix of **bold** and *italic* in one line",
            "Paragraph one\n\nParagraph two",
        ];
        fixtures.forEach((fixture) => {
            expect(markdownRoundTrip(fixture)).toBeTruthy();
        });
    });

    test("converts markdown to editable html and back", () => {
        const markdown = "- one\n- two with **bold**";
        const html = markdownToEditableHtml(markdown);
        const roundTripped = editableHtmlToMarkdown(html);
        expect(roundTripped).toMatch(/-\s+one/);
        expect(roundTripped).toContain("**bold**");
    });

    test("normalizes escaped and duplicate list markers", () => {
        const markdown = "- \\- first\n    - - second\n\t- \\- third";
        const html = markdownToEditableHtml(markdown);
        const roundTripped = editableHtmlToMarkdown(html);
        expect(roundTripped).toContain("- first");
        expect(roundTripped).toContain("- second");
        expect(roundTripped).toContain("- third");
        expect(roundTripped).not.toContain("\\-");
        expect(roundTripped).not.toContain("- -");
    });

    test("preserves two-space indentation in markdown normalization", () => {
        const markdown = "- parent\n  - child";
        const roundTripped = markdownRoundTrip(markdown);
        expect(roundTripped).toContain("- parent");
        expect(roundTripped).toMatch(/\n\s{2,}- child/);
    });

    test("indents and outdents list item via tab helper", () => {
        document.body.innerHTML = `
            <div id="root" contenteditable="true">
                <ul>
                    <li>first</li>
                    <li>second</li>
                </ul>
            </div>
        `;
        const root = document.getElementById("root");
        const second = root.querySelectorAll("li")[1];
        const textNode = second.firstChild;
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(textNode, 0);
        range.setEnd(textNode, textNode.textContent.length);
        selection.removeAllRanges();
        selection.addRange(range);

        expect(handleListTabIndent(root, false)).toBe(true);
        expect(root.querySelector("li ul li").textContent).toContain("second");

        const nested = root.querySelector("li ul li");
        const nestedText = nested.firstChild;
        const outdentRange = document.createRange();
        outdentRange.setStart(nestedText, 0);
        outdentRange.setEnd(nestedText, nestedText.textContent.length);
        selection.removeAllRanges();
        selection.addRange(outdentRange);

        expect(handleListTabIndent(root, true)).toBe(true);
        expect(root.querySelectorAll("ul > li").length).toBe(2);
    });

    test("outdent keeps relative order by moving trailing siblings under moved line", () => {
        document.body.innerHTML = `
            <div id="root" contenteditable="true">
                <ul>
                    <li>
                        top
                        <ul>
                            <li>first</li>
                            <li>second</li>
                            <li>third</li>
                        </ul>
                    </li>
                </ul>
            </div>
        `;
        const root = document.getElementById("root");
        const first = root.querySelector("ul ul li");
        const textNode = first.firstChild;
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(textNode, 0);
        range.setEnd(textNode, textNode.textContent.length);
        selection.removeAllRanges();
        selection.addRange(range);

        expect(handleListTabIndent(root, true)).toBe(true);
        const topLevelItems = root.querySelectorAll(":scope > ul > li");
        expect(topLevelItems.length).toBe(2);
        expect(topLevelItems[1].firstChild.textContent.trim()).toBe("first");
        const nestedTexts = Array.from(topLevelItems[1].querySelectorAll(":scope > ul > li")).map(
            (node) => node.textContent.trim()
        );
        expect(nestedTexts).toEqual(["second", "third"]);
    });

    test("indenting a line with child bullets only indents that line", () => {
        document.body.innerHTML = `
            <div id="root" contenteditable="true">
                <ul>
                    <li>anchor</li>
                    <li>
                        parent
                        <ul>
                            <li>child-a</li>
                            <li>child-b</li>
                        </ul>
                    </li>
                </ul>
            </div>
        `;
        const root = document.getElementById("root");
        const parent = root.querySelectorAll(":scope > ul > li")[1];
        const textNode = parent.firstChild;
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(textNode, 0);
        range.setEnd(textNode, textNode.textContent.length);
        selection.removeAllRanges();
        selection.addRange(range);

        expect(handleListTabIndent(root, false)).toBe(true);
        const top = root.querySelector(":scope > ul");
        const topLevel = Array.from(top.children).map((li) => li.firstChild.textContent.trim());
        expect(topLevel).toEqual(["anchor"]);
        const nestedUnderAnchor = root.querySelector(":scope > ul > li > ul");
        const nestedTexts = Array.from(nestedUnderAnchor.children).map((li) =>
            li.firstChild.textContent.trim()
        );
        expect(nestedTexts).toEqual(["parent", "child-a", "child-b"]);
    });
});
