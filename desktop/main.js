const path = require("path");
const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const { getSecret, setSecret, deleteSecret } = require("./services/keychain");
const { SidecarManager } = require("./services/sidecarManager");

const APP_NAME = "NeSyPaperGraph";
const OPENAI_KEY_SERVICE = `${APP_NAME}.desktop`;
const OPENAI_KEY_ACCOUNT = "OPENAI_API_KEY";
const APP_ACCESS_KEY_ACCOUNT = "APP_ACCESS_KEY";

let mainWindow = null;
let sidecar = null;
let runtimeState = {
    apiBaseUrl: "",
    sidecarPort: null,
    diagnostics: null,
};

function buildSecretEnvOverrides(openaiKey, appAccessKey) {
    const overrides = {};
    if (openaiKey && String(openaiKey).trim()) {
        overrides.OPENAI_API_KEY = String(openaiKey).trim();
    }
    if (appAccessKey && String(appAccessKey).trim()) {
        overrides.APP_ACCESS_KEY = String(appAccessKey).trim();
    }
    return overrides;
}

function getRendererEntry() {
    if (process.env.ELECTRON_RENDERER_URL) {
        return {
            type: "url",
            value: process.env.ELECTRON_RENDERER_URL,
        };
    }
    return {
        type: "file",
        value: path.join(__dirname, "..", "frontend", "build", "index.html"),
    };
}

function buildDesktopConfig() {
    return {
        isDesktop: true,
        apiBaseUrl: runtimeState.apiBaseUrl,
    };
}

async function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 1100,
        minHeight: 700,
        title: APP_NAME,
        webPreferences: {
            preload: path.join(__dirname, "preload.js"),
            contextIsolation: true,
            nodeIntegration: false,
            sandbox: true,
        },
    });

    const entry = getRendererEntry();
    if (entry.type === "url") {
        await mainWindow.loadURL(entry.value);
    } else {
        await mainWindow.loadFile(entry.value);
    }
}

function registerIpcHandlers() {
    ipcMain.handle("desktop:get-config", () => buildDesktopConfig());
    ipcMain.handle("desktop:get-runtime-state", () => runtimeState);

    ipcMain.handle("desktop:get-secret", async (_event, keyName) => {
        if (keyName === "OPENAI_API_KEY") {
            const value = await getSecret(OPENAI_KEY_SERVICE, OPENAI_KEY_ACCOUNT);
            return value || "";
        }
        if (keyName === "APP_ACCESS_KEY") {
            const value = await getSecret(OPENAI_KEY_SERVICE, APP_ACCESS_KEY_ACCOUNT);
            return value || "";
        }
        return "";
    });

    ipcMain.handle("desktop:set-secret", async (_event, keyName, value) => {
        const normalized = String(value || "").trim();
        if (keyName === "OPENAI_API_KEY") {
            if (!normalized) {
                await deleteSecret(OPENAI_KEY_SERVICE, OPENAI_KEY_ACCOUNT);
            } else {
                await setSecret(OPENAI_KEY_SERVICE, OPENAI_KEY_ACCOUNT, normalized);
            }
            await restartSidecarWithCurrentSecrets();
            return true;
        }
        if (keyName === "APP_ACCESS_KEY") {
            if (!normalized) {
                await deleteSecret(OPENAI_KEY_SERVICE, APP_ACCESS_KEY_ACCOUNT);
            } else {
                await setSecret(OPENAI_KEY_SERVICE, APP_ACCESS_KEY_ACCOUNT, normalized);
            }
            await restartSidecarWithCurrentSecrets();
            return true;
        }
        return false;
    });
}

async function restartSidecarWithCurrentSecrets() {
    const openaiKey = await getSecret(OPENAI_KEY_SERVICE, OPENAI_KEY_ACCOUNT);
    const appAccessKey = await getSecret(OPENAI_KEY_SERVICE, APP_ACCESS_KEY_ACCOUNT);
    runtimeState = await sidecar.restart(
        buildSecretEnvOverrides(openaiKey, appAccessKey)
    );
}

async function startSidecar() {
    const openaiKey = await getSecret(OPENAI_KEY_SERVICE, OPENAI_KEY_ACCOUNT);
    const appAccessKey = await getSecret(OPENAI_KEY_SERVICE, APP_ACCESS_KEY_ACCOUNT);
    sidecar = new SidecarManager({
        app,
        appName: APP_NAME,
        envOverrides: buildSecretEnvOverrides(openaiKey, appAccessKey),
    });

    runtimeState = await sidecar.start();
}

app.whenReady().then(async () => {
    registerIpcHandlers();
    try {
        await startSidecar();
    } catch (error) {
        await dialog.showMessageBox({
            type: "error",
            title: "Failed to start backend",
            message: "NeSyPaperGraph could not start its local backend.",
            detail: String(error?.message || error),
        });
        app.quit();
        return;
    }
    await createMainWindow();
});

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});

app.on("before-quit", async () => {
    if (sidecar) {
        await sidecar.stop();
    }
});
