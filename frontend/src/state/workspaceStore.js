import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

const STORAGE_KEY = "nesy_topic_workspace_state_v1";
const WORKSPACE_STATE_PATH = "/api/workspace/state";
const PERSIST_DEBOUNCE_MS = 800;
const MAX_READING_ITEMS = 200;
const MAX_THEME_NOTES = 150;
const MAX_ANNOTATIONS = 400;
const MAX_NOTE_CHARS = 12000;
const MAX_QUICK_NOTE_CHARS = 2000;

const initialState = {
    readingItems: [],
    themeNotes: [],
    paperAnnotations: {},
};

function createId(prefix) {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function reducer(state, action) {
    switch (action.type) {
        case "HYDRATE":
            return {
                ...state,
                ...action.payload,
                readingItems: Array.isArray(action.payload?.readingItems)
                    ? action.payload.readingItems
                    : state.readingItems,
                themeNotes: Array.isArray(action.payload?.themeNotes)
                    ? action.payload.themeNotes
                    : state.themeNotes,
                paperAnnotations:
                    action.payload?.paperAnnotations &&
                    typeof action.payload.paperAnnotations === "object"
                        ? action.payload.paperAnnotations
                        : state.paperAnnotations,
            };
        case "ADD_READING_ITEM":
            return {
                ...state,
                readingItems: [action.payload, ...state.readingItems],
            };
        case "UPDATE_READING_ITEM":
            return {
                ...state,
                readingItems: state.readingItems.map((item) =>
                    item.id === action.payload.id
                        ? {
                              ...item,
                              ...action.payload.patch,
                              updatedAt: new Date().toISOString(),
                          }
                        : item
                ),
            };
        case "REMOVE_READING_ITEM":
            return {
                ...state,
                readingItems: state.readingItems.filter(
                    (item) => item.id !== action.payload.id
                ),
            };
        case "CREATE_THEME_NOTE":
            return {
                ...state,
                themeNotes: [action.payload, ...state.themeNotes],
            };
        case "UPDATE_THEME_NOTE":
            return {
                ...state,
                themeNotes: state.themeNotes.map((note) =>
                    note.id === action.payload.id
                        ? {
                              ...note,
                              ...action.payload.patch,
                              updatedAt: new Date().toISOString(),
                          }
                        : note
                ),
            };
        case "UPSERT_THEME_NOTE": {
            const now = new Date().toISOString();
            const existingNote = state.themeNotes.find(
                (note) => note.id === action.payload.id
            );
            const nextNote = {
                id: action.payload.id || createId("theme"),
                themeTitle: action.payload.themeTitle?.trim() || "",
                linkedPaperTitles: action.payload.linkedPaperTitles || [],
                sections: {
                        notes:
                            action.payload.sections?.notes ||
                            [
                                action.payload.sections?.summary,
                                action.payload.sections?.evidence,
                                action.payload.sections?.questions,
                            ]
                                .filter(Boolean)
                                .join("\n\n"),
                        toRead:
                            action.payload.sections?.toRead ||
                            action.payload.sections?.nextReads ||
                            "",
                },
                createdAt: existingNote?.createdAt || now,
                updatedAt: now,
            };

            if (!existingNote) {
                return {
                    ...state,
                    themeNotes: [nextNote, ...state.themeNotes],
                };
            }

            return {
                ...state,
                themeNotes: state.themeNotes.map((note) =>
                    note.id === nextNote.id ? nextNote : note
                ),
            };
        }
        case "UPSERT_PAPER_ANNOTATION":
            return {
                ...state,
                paperAnnotations: {
                    ...state.paperAnnotations,
                    [action.payload.paperTitle]: {
                        paperTitle: action.payload.paperTitle,
                        ...(state.paperAnnotations[action.payload.paperTitle] || {}),
                        ...action.payload.patch,
                        updatedAt: new Date().toISOString(),
                    },
                },
            };
        case "LINK_PAPER_TO_THEME":
            return {
                ...state,
                themeNotes: state.themeNotes.map((note) => {
                    if (note.id !== action.payload.noteId) return note;
                    const linkedPaperTitles = Array.from(
                        new Set([
                            ...(note.linkedPaperTitles || []),
                            action.payload.paperTitle,
                        ])
                    );
                    return {
                        ...note,
                        linkedPaperTitles,
                        updatedAt: new Date().toISOString(),
                    };
                }),
            };
        case "SET_PAPER_THEME_MEMBERSHIP": {
            const selectedThemeIds = new Set(action.payload.themeIds || []);
            const paperTitle = action.payload.paperTitle;
            return {
                ...state,
                themeNotes: state.themeNotes.map((note) => {
                    const existingTitles = new Set(note.linkedPaperTitles || []);
                    if (selectedThemeIds.has(note.id)) {
                        existingTitles.add(paperTitle);
                    } else {
                        existingTitles.delete(paperTitle);
                    }
                    return {
                        ...note,
                        linkedPaperTitles: Array.from(existingTitles),
                        updatedAt: new Date().toISOString(),
                    };
                }),
            };
        }
        default:
            return state;
    }
}

function parseLocalState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return null;
        return JSON.parse(raw);
    } catch (error) {
        console.error("Failed to parse topic workspace local state:", error);
        return null;
    }
}

function clampText(value, maxChars) {
    const normalized = typeof value === "string" ? value : "";
    return normalized.slice(0, maxChars);
}

function trimWorkspaceState(state) {
    const safeState = state || {};
    const readingItems = Array.isArray(safeState.readingItems)
        ? safeState.readingItems.slice(0, MAX_READING_ITEMS).map((item) => ({
              ...item,
              quickNote: clampText(item.quickNote, MAX_QUICK_NOTE_CHARS),
              authors: Array.isArray(item.authors) ? item.authors : [],
          }))
        : [];
    const themeNotes = Array.isArray(safeState.themeNotes)
        ? safeState.themeNotes.slice(0, MAX_THEME_NOTES).map((note) => ({
              ...note,
              sections: {
                  notes: clampText(note?.sections?.notes, MAX_NOTE_CHARS),
                  toRead: clampText(note?.sections?.toRead, MAX_NOTE_CHARS),
              },
          }))
        : [];
    const annotationEntries = Object.entries(safeState.paperAnnotations || {}).slice(
        0,
        MAX_ANNOTATIONS
    );
    const paperAnnotations = Object.fromEntries(
        annotationEntries.map(([title, annotation]) => [
            title,
            {
                ...annotation,
                notesMarkdown: clampText(
                    annotation?.notesMarkdown,
                    MAX_NOTE_CHARS
                ),
            },
        ])
    );
    return {
        readingItems,
        themeNotes,
        paperAnnotations,
    };
}

function persistLocalState(state) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(trimWorkspaceState(state)));
    } catch (error) {
        console.error("Failed to persist topic workspace state:", error);
    }
}

export function useWorkspaceStore(options = {}) {
    const { apiBase = "", apiFetch = null, isEnabled = true } = options;
    const [state, dispatch] = useReducer(reducer, initialState);
    const [syncMode, setSyncMode] = useState("local");
    const [syncWarning, setSyncWarning] = useState("");
    const [hasHydrated, setHasHydrated] = useState(false);
    const fallbackModeRef = useRef(false);

    useEffect(() => {
        let isCancelled = false;
        const parsed = parseLocalState();
        if (parsed) {
            dispatch({ type: "HYDRATE", payload: trimWorkspaceState(parsed) });
        }

        const canUseApi = Boolean(apiBase && apiFetch && isEnabled);
        if (!canUseApi) {
            fallbackModeRef.current = true;
            setSyncMode("local");
            setHasHydrated(true);
            return () => {
                isCancelled = true;
            };
        }

        fallbackModeRef.current = false;
        const hydrateFromServer = async () => {
            try {
                const response = await apiFetch(`${apiBase}${WORKSPACE_STATE_PATH}`);
                if (!response.ok) {
                    throw new Error(`Workspace state fetch failed: ${response.status}`);
                }
                const payload = await response.json();
                if (isCancelled) return;
                const normalizedPayload = trimWorkspaceState(payload);
                dispatch({ type: "HYDRATE", payload: normalizedPayload });
                persistLocalState(normalizedPayload);
                setSyncMode("remote");
                setSyncWarning("");
            } catch (error) {
                if (isCancelled) return;
                fallbackModeRef.current = true;
                setSyncMode("local-fallback");
                setSyncWarning(
                    "Workspace API unavailable. Using local-only workspace cache."
                );
                console.warn("Workspace API hydrate failed, using local cache:", error);
            } finally {
                if (!isCancelled) {
                    setHasHydrated(true);
                }
            }
        };

        hydrateFromServer();
        return () => {
            isCancelled = true;
        };
    }, [apiBase, apiFetch, isEnabled]);

    useEffect(() => {
        if (!hasHydrated) return;
        persistLocalState(state);

        const canSyncRemote =
            Boolean(apiBase && apiFetch && isEnabled) && !fallbackModeRef.current;
        if (!canSyncRemote) return;

        const timer = setTimeout(async () => {
            try {
                const response = await apiFetch(`${apiBase}${WORKSPACE_STATE_PATH}`, {
                    method: "PUT",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(trimWorkspaceState(state)),
                });
                if (!response.ok) {
                    throw new Error(`Workspace state save failed: ${response.status}`);
                }
            } catch (error) {
                fallbackModeRef.current = true;
                setSyncMode("local-fallback");
                setSyncWarning(
                    "Workspace API unavailable. Continuing in local-only mode."
                );
                console.warn("Workspace API save failed, staying local:", error);
            }
        }, PERSIST_DEBOUNCE_MS);

        return () => clearTimeout(timer);
    }, [state, apiBase, apiFetch, isEnabled, hasHydrated]);

    const actions = useMemo(
        () => ({
            addReadingItem(item) {
                dispatch({
                    type: "ADD_READING_ITEM",
                    payload: {
                        id: createId("read"),
                        sourceType: item.sourceType || "url",
                        status: item.status || "inbox",
                        topicHints: item.topicHints || [],
                        linkedPaperTitle: item.linkedPaperTitle || null,
                        linkedThemeId: item.linkedThemeId || null,
                        title: item.title || "",
                        url: item.url || "",
                        semanticScholarPaperId: item.semanticScholarPaperId || null,
                        authors: Array.isArray(item.authors) ? item.authors : [],
                        year:
                            typeof item.year === "number" && Number.isFinite(item.year)
                                ? item.year
                                : null,
                        venue: item.venue || null,
                        quickNote: item.quickNote || "",
                        createdAt: new Date().toISOString(),
                        updatedAt: new Date().toISOString(),
                    },
                });
            },
            updateReadingItem(id, patch) {
                dispatch({ type: "UPDATE_READING_ITEM", payload: { id, patch } });
            },
            removeReadingItem(id) {
                dispatch({ type: "REMOVE_READING_ITEM", payload: { id } });
            },
            createThemeNote(input) {
                const themeTitle = input.themeTitle?.trim() || "Untitled Theme";
                const note = {
                    id: createId("theme"),
                    themeTitle,
                    linkedPaperTitles: input.linkedPaperTitles || [],
                    sections: {
                        notes:
                            input.sections?.notes ||
                            [
                                input.sections?.summary,
                                input.sections?.evidence,
                                input.sections?.questions,
                            ]
                                .filter(Boolean)
                                .join("\n\n"),
                        toRead:
                            input.sections?.toRead || input.sections?.nextReads || "",
                    },
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString(),
                };
                dispatch({ type: "CREATE_THEME_NOTE", payload: note });
                return note;
            },
            updateThemeNote(id, patch) {
                dispatch({ type: "UPDATE_THEME_NOTE", payload: { id, patch } });
            },
            upsertThemeNote(themeNoteInput) {
                const normalizedInput = {
                    ...themeNoteInput,
                    id: themeNoteInput.id || createId("theme"),
                };
                dispatch({
                    type: "UPSERT_THEME_NOTE",
                    payload: normalizedInput,
                });
                return normalizedInput;
            },
            upsertPaperAnnotation(paperTitle, patch) {
                dispatch({
                    type: "UPSERT_PAPER_ANNOTATION",
                    payload: { paperTitle, patch },
                });
            },
            setPaperThemeMembership(paperTitle, themeIds) {
                dispatch({
                    type: "SET_PAPER_THEME_MEMBERSHIP",
                    payload: { paperTitle, themeIds },
                });
            },
            linkPaperToTheme(noteId, paperTitle) {
                dispatch({
                    type: "LINK_PAPER_TO_THEME",
                    payload: { noteId, paperTitle },
                });
            },
        }),
        []
    );

    const selectors = useMemo(
        () => ({
            getPaperAnnotation(paperTitle) {
                return (
                    state.paperAnnotations[paperTitle] || {
                        paperTitle,
                        notesMarkdown: "",
                        topicLinks: [],
                        status: "unread",
                    }
                );
            },
            getThemeById(noteId) {
                return state.themeNotes.find((note) => note.id === noteId) || null;
            },
        }),
        [state.paperAnnotations, state.themeNotes]
    );

    const clearLocalWorkspaceState = useCallback(() => {
        localStorage.removeItem(STORAGE_KEY);
        window.location.reload();
    }, []);

    return {
        state,
        actions,
        selectors,
        clearLocalWorkspaceState,
        syncMode,
        syncWarning,
    };
}
