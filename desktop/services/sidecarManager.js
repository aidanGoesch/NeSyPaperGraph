const fs = require("fs");
const net = require("net");
const path = require("path");
const { spawn } = require("child_process");

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function createHttpGetJson() {
    return async function getJson(url) {
        const response = await fetch(url, {
            method: "GET",
            cache: "no-store",
        });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }
        return response.json();
    };
}

async function findOpenPort() {
    return new Promise((resolve, reject) => {
        const server = net.createServer();
        server.unref();
        server.on("error", reject);
        server.listen(0, "127.0.0.1", () => {
            const address = server.address();
            const port = typeof address === "object" && address ? address.port : 0;
            server.close(() => resolve(port));
        });
    });
}

function resolvePaths(app) {
    const rootDir = path.join(__dirname, "..", "..");
    const backendDir = path.join(rootDir, "backend");
    const packagedSidecar = path.join(process.resourcesPath, "sidecar", "nesy-backend");
    return {
        rootDir,
        backendDir,
        packagedSidecar,
        backendVenvPython: path.join(backendDir, ".venv", "bin", "python"),
        logFile: path.join(app.getPath("logs"), "sidecar.log"),
        dataDir: path.join(app.getPath("userData"), "data"),
    };
}

class SidecarManager {
    constructor({ app, appName, envOverrides = {} }) {
        this.app = app;
        this.appName = appName;
        this.envOverrides = envOverrides;
        this.child = null;
        this.port = null;
        this.paths = resolvePaths(app);
        this.getJson = createHttpGetJson();
        this.stopping = false;
        this.lastExtraEnv = {};
    }

    async start(extraEnv = {}) {
        this.lastExtraEnv = extraEnv;
        fs.mkdirSync(path.dirname(this.paths.logFile), { recursive: true });
        fs.mkdirSync(this.paths.dataDir, { recursive: true });

        this.port = this.port || (await findOpenPort());
        const env = {
            ...process.env,
            ...this.envOverrides,
            ...extraEnv,
            DESKTOP_APP_MODE: "true",
            FRONTEND_URL: process.env.FRONTEND_URL || "http://localhost:3000,null",
            LOCAL_DATA_DIR: process.env.LOCAL_DATA_DIR || this.paths.dataDir,
            PORT: String(this.port),
        };

        const child = this.spawnSidecarProcess(env);
        this.child = child;
        this.bindLogs(child);
        child.once("close", async () => {
            if (this.stopping || this.child !== child) return;
            this.child = null;
            try {
                await this.start(this.lastExtraEnv);
            } catch (_error) {
                // next API call will surface availability failure in renderer
            }
        });
        await this.waitForHealthy();
        const diagnostics = await this.safeDiagnostics();
        return {
            apiBaseUrl: `http://127.0.0.1:${this.port}`,
            sidecarPort: this.port,
            diagnostics,
        };
    }

    spawnSidecarProcess(env) {
        if (this.app.isPackaged && fs.existsSync(this.paths.packagedSidecar)) {
            return spawn(this.paths.packagedSidecar, ["--port", String(this.port)], {
                cwd: process.resourcesPath,
                env,
                stdio: ["ignore", "pipe", "pipe"],
            });
        }

        const pythonExecutable = fs.existsSync(this.paths.backendVenvPython)
            ? this.paths.backendVenvPython
            : "python3";
        return spawn(
            pythonExecutable,
            ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", String(this.port)],
            {
                cwd: this.paths.backendDir,
                env,
                stdio: ["ignore", "pipe", "pipe"],
            }
        );
    }

    bindLogs(child) {
        const stream = fs.createWriteStream(this.paths.logFile, { flags: "a" });
        const prefix = `[${new Date().toISOString()}] `;
        stream.write(`${prefix}Starting sidecar on port ${this.port}\n`);

        child.stdout?.on("data", (chunk) => {
            stream.write(String(chunk));
        });
        child.stderr?.on("data", (chunk) => {
            stream.write(String(chunk));
        });
        child.on("close", (code, signal) => {
            stream.write(`Sidecar exited code=${code} signal=${signal}\n`);
            stream.end();
        });
    }

    async waitForHealthy(timeoutMs = 45000) {
        const healthUrl = `http://127.0.0.1:${this.port}/health`;
        const startedAt = Date.now();

        while (Date.now() - startedAt < timeoutMs) {
            if (!this.child || this.child.exitCode !== null) {
                throw new Error("Sidecar process exited before becoming healthy.");
            }
            try {
                const payload = await this.getJson(healthUrl);
                if (payload?.status === "ok") {
                    return;
                }
            } catch (_error) {
                // retry until timeout
            }
            await sleep(500);
        }
        throw new Error("Timed out waiting for local backend health.");
    }

    async safeDiagnostics() {
        try {
            return await this.getJson(
                `http://127.0.0.1:${this.port}/api/runtime/diagnostics`
            );
        } catch (_error) {
            return null;
        }
    }

    async stop() {
        if (!this.child) return;
        const child = this.child;
        this.child = null;
        this.stopping = true;
        await new Promise((resolve) => {
            let settled = false;
            const done = () => {
                if (!settled) {
                    settled = true;
                    resolve();
                }
            };
            child.once("close", done);
            child.kill("SIGTERM");
            setTimeout(() => {
                if (!settled) {
                    child.kill("SIGKILL");
                    done();
                }
            }, 5000);
        });
        this.stopping = false;
    }

    async restart(extraEnv = {}) {
        await this.stop();
        return this.start(extraEnv);
    }
}

module.exports = {
    SidecarManager,
};
