import React, { useEffect, useRef, useState } from "react";
import {
    editableHtmlToMarkdown,
    handleListTabIndent,
    markdownToEditableHtml,
} from "./paperNotesRichText";

export default function PaperNotesEditor({
    id,
    markdownValue,
    onMarkdownChange,
    onError,
    maxChars,
    placeholder,
}) {
    const editorRef = useRef(null);
    const lastRenderedMarkdownRef = useRef(String(markdownValue || ""));
    const [fallbackPlaintextMode, setFallbackPlaintextMode] = useState(false);
    const [fallbackValue, setFallbackValue] = useState(String(markdownValue || ""));

    const applyMarkdownToEditor = (nextMarkdown) => {
        const editor = editorRef.current;
        if (!editor) return;
        try {
            const html = markdownToEditableHtml(nextMarkdown);
            editor.innerHTML = html;
            setFallbackPlaintextMode(false);
            onError("");
        } catch (error) {
            // Safety-first fallback: keep raw text editable instead of risking data corruption.
            setFallbackPlaintextMode(true);
            setFallbackValue(String(nextMarkdown || ""));
            onError(
                "Rich notes formatting failed for this note. Showing plain text mode to protect content."
            );
        }
    };

    useEffect(() => {
        const normalized = String(markdownValue || "");
        if (fallbackPlaintextMode) {
            setFallbackValue(normalized);
            lastRenderedMarkdownRef.current = normalized;
            return;
        }
        if (normalized === lastRenderedMarkdownRef.current) return;
        applyMarkdownToEditor(normalized);
        lastRenderedMarkdownRef.current = normalized;
    }, [fallbackPlaintextMode, markdownValue]);

    useEffect(() => {
        if (fallbackPlaintextMode) return;
        applyMarkdownToEditor(String(markdownValue || ""));
    }, []);

    const commitFromEditor = () => {
        const editor = editorRef.current;
        if (!editor) return;
        try {
            const nextMarkdown = editableHtmlToMarkdown(editor.innerHTML);
            if (nextMarkdown.length > maxChars) {
                onError(`Notes are limited to ${maxChars.toLocaleString()} characters.`);
                applyMarkdownToEditor(lastRenderedMarkdownRef.current);
                return;
            }
            lastRenderedMarkdownRef.current = nextMarkdown;
            onError("");
            onMarkdownChange(nextMarkdown);
        } catch (error) {
            setFallbackPlaintextMode(true);
            setFallbackValue(lastRenderedMarkdownRef.current);
            onError(
                "Could not convert rich text safely. Switched to plain text mode to preserve notes."
            );
        }
    };

    const tryAutoStartBullet = (event) => {
        if (event.key !== " " || event.shiftKey || event.altKey || event.metaKey || event.ctrlKey) {
            return false;
        }
        const selection = window.getSelection?.();
        if (!selection || selection.rangeCount === 0 || !selection.isCollapsed) return false;
        const range = selection.getRangeAt(0);
        const editor = editorRef.current;
        if (!editor || !editor.contains(range.endContainer)) return false;

        let block = range.endContainer;
        while (block && block !== editor) {
            if (
                block.nodeType === Node.ELEMENT_NODE &&
                ["P", "DIV", "LI"].includes(block.tagName)
            ) {
                break;
            }
            block = block.parentNode;
        }
        if (!block || block === editor || block.tagName === "LI") return false;

        const textBeforeRange = document.createRange();
        textBeforeRange.selectNodeContents(block);
        textBeforeRange.setEnd(range.endContainer, range.endOffset);
        const textBeforeCursor = textBeforeRange.toString();
        if (!/^\s*-$/.test(textBeforeCursor)) return false;

        event.preventDefault();
        block.textContent = "";
        if (typeof document.execCommand === "function") {
            document.execCommand("insertUnorderedList", false);
        }
        commitFromEditor();
        return true;
    };

    const handleKeyDown = (event) => {
        if (fallbackPlaintextMode) return;
        if (tryAutoStartBullet(event)) return;
        const isPrimaryModifier = event.metaKey || event.ctrlKey;
        const key = String(event.key || "").toLowerCase();

        if (isPrimaryModifier && (key === "b" || key === "i")) {
            event.preventDefault();
            const command = key === "b" ? "bold" : "italic";
            if (typeof document.execCommand === "function") {
                document.execCommand(command, false);
            }
            commitFromEditor();
            return;
        }

        if (event.key === "Tab") {
            event.preventDefault();
            const handled = handleListTabIndent(editorRef.current, event.shiftKey);
            if (handled) {
                commitFromEditor();
            }
        }
    };

    const handlePaste = (event) => {
        const clipboardItems = Array.from(event.clipboardData?.items || []);
        const hasImage = clipboardItems.some(
            (item) => item.kind === "file" && item.type.startsWith("image/")
        );
        if (!hasImage) return;
        event.preventDefault();
        onError("Image pasting is disabled in notes. Please paste text only.");
    };

    if (fallbackPlaintextMode) {
        return (
            <textarea
                id={id}
                className="paper-note-modal-textarea"
                value={fallbackValue}
                onChange={(event) => {
                    const next = event.target.value;
                    if (next.length > maxChars) {
                        onError(`Notes are limited to ${maxChars.toLocaleString()} characters.`);
                        return;
                    }
                    setFallbackValue(next);
                    lastRenderedMarkdownRef.current = next;
                    onError("");
                    onMarkdownChange(next);
                }}
                onPaste={handlePaste}
                placeholder={placeholder}
            />
        );
    }

    return (
        <div
            id={id}
            ref={editorRef}
            className="paper-note-rich-editor"
            contentEditable
            suppressContentEditableWarning
            role="textbox"
            aria-label="Paper notes editor"
            aria-multiline="true"
            data-placeholder={placeholder}
            onInput={commitFromEditor}
            onBlur={commitFromEditor}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
        />
    );
}
