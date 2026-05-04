const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("desktopBridge", {
    getConfig: () => ipcRenderer.invoke("desktop:get-config"),
    getRuntimeState: () => ipcRenderer.invoke("desktop:get-runtime-state"),
    getSecret: (keyName) => ipcRenderer.invoke("desktop:get-secret", keyName),
    setSecret: (keyName, value) =>
        ipcRenderer.invoke("desktop:set-secret", keyName, value),
    onOpenInReaderUrl: (handler) => {
        if (typeof handler !== "function") {
            return () => {};
        }
        const listener = (_event, url) => handler(url);
        ipcRenderer.on("desktop:open-in-reader-url", listener);
        return () => {
            ipcRenderer.removeListener("desktop:open-in-reader-url", listener);
        };
    },
});
