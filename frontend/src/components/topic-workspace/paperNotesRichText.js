import TurndownService from "turndown";

const PAPER_NOTES_BACKUP_KEY = "nesy_paper_notes_backup_v1";
const PAPER_NOTES_ROLLOUT_KEY = "nesy_paper_notes_rich_editor_enabled";

const ALLOWED_TAGS = new Set([
    "P",
    "BR",
    "STRONG",
    "EM",
    "B",
    "I",
    "UL",
    "OL",
    "LI",
    "BLOCKQUOTE",
    "CODE",
    "PRE",
    "S",
    "DEL",
    "SPAN",
    "DIV",
]);

const turndown = new TurndownService({
    bulletListMarker: "-",
    emDelimiter: "*",
    strongDelimiter: "**",
    codeBlockStyle: "fenced",
});

function createHtmlDoc() {
    if (typeof document !== "undefined" && document.implementation) {
        return document.implementation.createHTMLDocument("");
    }
    if (typeof DOMParser !== "undefined") {
        const parser = new DOMParser();
        return parser.parseFromString("<!doctype html><html><body></body></html>", "text/html");
    }
    throw new Error("HTML document APIs are unavailable in this environment.");
}

function sanitizeNode(node, doc) {
    if (!node) return null;
    if (node.nodeType === Node.TEXT_NODE) {
        return doc.createTextNode(node.textContent || "");
    }
    if (node.nodeType !== Node.ELEMENT_NODE) {
        return null;
    }

    const tag = (node.tagName || "").toUpperCase();
    if (!ALLOWED_TAGS.has(tag)) {
        const fragment = doc.createDocumentFragment();
        Array.from(node.childNodes || []).forEach((child) => {
            const cleanChild = sanitizeNode(child, doc);
            if (cleanChild) fragment.appendChild(cleanChild);
        });
        return fragment;
    }

    const nextTag =
        tag === "B" ? "strong" : tag === "I" ? "em" : tag === "DEL" ? "s" : tag.toLowerCase();
    const clean = doc.createElement(nextTag);
    Array.from(node.childNodes || []).forEach((child) => {
        const cleanChild = sanitizeNode(child, doc);
        if (cleanChild) clean.appendChild(cleanChild);
    });
    return clean;
}

function sanitizeHtml(html) {
    const parser = new DOMParser();
    const parsed = parser.parseFromString(`<div>${html || ""}</div>`, "text/html");
    const wrapper = parsed.body.firstElementChild;
    const cleanDoc = createHtmlDoc();
    const cleanRoot = cleanDoc.createElement("div");

    if (wrapper) {
        Array.from(wrapper.childNodes).forEach((child) => {
            const cleanChild = sanitizeNode(child, cleanDoc);
            if (cleanChild) cleanRoot.appendChild(cleanChild);
        });
    }

    if (!cleanRoot.textContent?.trim() && cleanRoot.querySelectorAll("li").length === 0) {
        cleanRoot.innerHTML = "<p><br></p>";
    }
    return cleanRoot.innerHTML;
}

function normalizeMarkdown(markdown) {
    const cleaned = String(markdown || "")
        .replace(/\r\n/g, "\n")
        .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, "")
        .replace(/[ \t]+\n/g, "\n")
        .trimEnd();
    return normalizeBulletLines(cleaned);
}

function normalizeBulletLines(markdown) {
    const lines = String(markdown || "").split("\n");
    return lines
        .map((line) => {
            const expanded = line.replace(/\t/g, "    ");
            const listMatch = expanded.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
            if (!listMatch) {
                return expanded.replace(/^(\s*)\\-\s+/, "$1- ");
            }
            const [, indent, marker, rawContent] = listMatch;
            const normalizedIndent = indent;
            let content = rawContent;
            content = content.replace(/^\\-\s+/, "");
            content = content.replace(/^[-*+]\s+/, "");
            content = content.trimEnd();
            return `${normalizedIndent}${marker} ${content}`;
        })
        .join("\n");
}

export function isRichPaperNotesEditorEnabled() {
    if (typeof window === "undefined" || !window.localStorage) return true;
    return window.localStorage.getItem(PAPER_NOTES_ROLLOUT_KEY) !== "0";
}

export function backupPaperNoteSnapshot(annotationKey, notesMarkdown) {
    if (!annotationKey || typeof window === "undefined" || !window.localStorage) return;
    const nextEntry = {
        notesMarkdown: String(notesMarkdown || ""),
        updatedAt: new Date().toISOString(),
    };
    try {
        const raw = window.localStorage.getItem(PAPER_NOTES_BACKUP_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        const previous = parsed?.[annotationKey];
        if (previous?.notesMarkdown === nextEntry.notesMarkdown) return;

        const next = {
            ...(parsed && typeof parsed === "object" ? parsed : {}),
            [annotationKey]: nextEntry,
        };
        window.localStorage.setItem(PAPER_NOTES_BACKUP_KEY, JSON.stringify(next));
    } catch (error) {
        console.warn("Unable to store paper notes backup snapshot.", error);
    }
}

function escapeHtml(value) {
    return String(value || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function renderInlineMarkdown(text) {
    const escaped = escapeHtml(text);
    return escaped
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/_([^_]+)_/g, "<em>$1</em>");
}

function createListElement(doc, marker) {
    if (/^\d+\.$/.test(marker || "")) return doc.createElement("ol");
    return doc.createElement("ul");
}

function appendMarkdownAsHtml(root, markdown) {
    const doc = root.ownerDocument;
    const lines = normalizeMarkdown(markdown).split("\n");
    let paragraphBuffer = [];
    let listStack = [];

    const flushParagraph = () => {
        if (paragraphBuffer.length === 0) return;
        const paragraph = doc.createElement("p");
        paragraph.innerHTML = renderInlineMarkdown(paragraphBuffer.join("\n"));
        root.appendChild(paragraph);
        paragraphBuffer = [];
    };

    const closeListsDownTo = (targetDepth) => {
        while (listStack.length > targetDepth) {
            listStack.pop();
        }
    };

    const appendListContinuationLine = (line) => {
        const currentList = listStack[listStack.length - 1]?.list || null;
        const currentItem = currentList?.lastElementChild || null;
        if (!currentItem) return false;
        currentItem.appendChild(doc.createElement("br"));
        const continuation = doc.createElement("span");
        continuation.innerHTML = renderInlineMarkdown(line.trim());
        currentItem.appendChild(continuation);
        return true;
    };

    lines.forEach((line) => {
        const listMatch = line.match(/^(\s*)([-*+]|\d+\.)\s+(.*)$/);
        if (!listMatch) {
            if (!line.trim()) {
                flushParagraph();
                closeListsDownTo(0);
            } else {
                if (listStack.length > 0 && /^\s+/.test(line)) {
                    if (appendListContinuationLine(line)) return;
                }
                flushParagraph();
                closeListsDownTo(0);
                paragraphBuffer.push(line);
            }
            return;
        }

        flushParagraph();
        const [, indent, marker, content] = listMatch;
        if (!content.trim()) {
            closeListsDownTo(0);
            paragraphBuffer.push("-");
            return;
        }
        const normalizedIndent = indent.replace(/\t/g, "    ");
        const indentLen = normalizedIndent.length;
        const depth =
            indentLen <= 0
                ? 1
                : Math.min(
                      4,
                      indentLen % 4 === 0
                          ? Math.floor(indentLen / 4) + 1
                          : Math.floor(indentLen / 2) + 1
                  );

        if (listStack.length > depth) {
            closeListsDownTo(depth);
        }

        while (listStack.length < depth) {
            const parentList = listStack[listStack.length - 1]?.list || null;
            const nextList = createListElement(doc, marker);
            if (!parentList) {
                root.appendChild(nextList);
            } else {
                const parentLi = parentList.lastElementChild || doc.createElement("li");
                if (!parentList.lastElementChild) parentList.appendChild(parentLi);
                parentLi.appendChild(nextList);
            }
            listStack.push({ list: nextList });
        }

        const currentList = listStack[listStack.length - 1].list;
        if (
            (currentList.tagName === "OL" && !/^\d+\.$/.test(marker)) ||
            (currentList.tagName === "UL" && /^\d+\.$/.test(marker))
        ) {
            closeListsDownTo(depth - 1);
            const parentList = listStack[listStack.length - 1]?.list || null;
            const nextList = createListElement(doc, marker);
            if (!parentList) {
                root.appendChild(nextList);
            } else {
                const parentLi = parentList.lastElementChild || doc.createElement("li");
                if (!parentList.lastElementChild) parentList.appendChild(parentLi);
                parentLi.appendChild(nextList);
            }
            listStack.push({ list: nextList });
        }

        const li = doc.createElement("li");
        li.innerHTML = renderInlineMarkdown(content);
        listStack[listStack.length - 1].list.appendChild(li);
    });

    flushParagraph();
    closeListsDownTo(0);
}

export function markdownToEditableHtml(markdown) {
    const doc = createHtmlDoc();
    const root = doc.createElement("div");
    appendMarkdownAsHtml(root, markdown);
    const rendered = root.innerHTML || "<p><br></p>";
    return sanitizeHtml(rendered);
}

export function editableHtmlToMarkdown(html) {
    const safeHtml = sanitizeHtml(String(html || ""));
    const rawMarkdown = turndown.turndown(safeHtml);
    return normalizeMarkdown(rawMarkdown);
}

export function markdownRoundTrip(markdown) {
    return editableHtmlToMarkdown(markdownToEditableHtml(markdown));
}

function getSelectionListItem(root) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;
    let node = selection.anchorNode;
    while (node && node !== root) {
        if (node.nodeType === Node.ELEMENT_NODE && node.tagName === "LI") {
            return node;
        }
        node = node.parentNode;
    }
    return null;
}

function ensureNestedList(li, listTagName) {
    const lastChild = li.lastElementChild;
    if (lastChild && lastChild.tagName === listTagName) {
        return lastChild;
    }
    const nested = li.ownerDocument.createElement(listTagName.toLowerCase());
    li.appendChild(nested);
    return nested;
}

function setCursorToEnd(element) {
    const selection = window.getSelection();
    if (!selection) return;
    const range = document.createRange();
    range.selectNodeContents(element);
    range.collapse(false);
    selection.removeAllRanges();
    selection.addRange(range);
}

function promoteDirectChildrenOneLevel(li) {
    const parentList = li.parentElement;
    if (!parentList) return;
    const childLists = Array.from(li.children).filter((child) =>
        child?.tagName ? ["UL", "OL"].includes(child.tagName) : false
    );
    if (childLists.length === 0) return;

    const anchor = li.nextSibling;
    const promoted = [];
    childLists.forEach((list) => {
        Array.from(list.children).forEach((candidate) => {
            if (candidate.tagName === "LI") {
                promoted.push(candidate);
            }
        });
        list.remove();
    });
    promoted.forEach((item) => {
        if (anchor) {
            parentList.insertBefore(item, anchor);
        } else {
            parentList.appendChild(item);
        }
    });
}

function indentListItem(li) {
    const parentList = li.parentElement;
    if (!parentList) return false;
    const previousItem = li.previousElementSibling;
    if (!previousItem || previousItem.tagName !== "LI") return true;
    const nestedList = ensureNestedList(previousItem, parentList.tagName);
    nestedList.appendChild(li);
    promoteDirectChildrenOneLevel(li);
    setCursorToEnd(li);
    return true;
}

function outdentListItem(li) {
    const parentList = li.parentElement;
    if (!parentList) return false;
    if (li.querySelector(":scope > ul > li, :scope > ol > li")) {
        // Keep child indentation unchanged by refusing to move parent items with nested bullets.
        return false;
    }
    const parentItem = parentList.parentElement;
    if (!parentItem || parentItem.tagName !== "LI") return true;
    const grandList = parentItem.parentElement;
    if (!grandList) return true;

    const trailingSiblings = [];
    let sibling = li.nextElementSibling;
    while (sibling) {
        const nextSibling = sibling.nextElementSibling;
        trailingSiblings.push(sibling);
        sibling = nextSibling;
    }
    if (trailingSiblings.length > 0) {
        const inheritedNestedList = li.ownerDocument.createElement(parentList.tagName.toLowerCase());
        trailingSiblings.forEach((item) => inheritedNestedList.appendChild(item));
        li.appendChild(inheritedNestedList);
    }

    if (parentItem.nextSibling) {
        grandList.insertBefore(li, parentItem.nextSibling);
    } else {
        grandList.appendChild(li);
    }
    if (!parentList.querySelector("li")) {
        parentList.remove();
    }
    setCursorToEnd(li);
    return true;
}

export function handleListTabIndent(root, isOutdent = false) {
    if (!root || typeof window === "undefined" || !window.getSelection) return false;
    const li = getSelectionListItem(root);
    if (!li) return false;
    return isOutdent ? outdentListItem(li) : indentListItem(li);
}
