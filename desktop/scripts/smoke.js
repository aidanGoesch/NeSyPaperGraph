const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

function run(command, args, options = {}) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, {
            stdio: "pipe",
            ...options,
        });
        let stderr = "";
        child.stderr.on("data", (chunk) => {
            stderr += String(chunk);
        });
        child.on("error", reject);
        child.on("close", (code) => {
            if (code === 0) {
                resolve();
                return;
            }
            reject(new Error(`${command} exited ${code}: ${stderr}`));
        });
    });
}

async function checkHealth(apiBase) {
    const response = await fetch(`${apiBase}/health`);
    if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
    }
    const payload = await response.json();
    if (payload.status !== "ok") {
        throw new Error("Health payload did not return status=ok");
    }
}

async function main() {
    const port = 8765;
    const apiBase = `http://127.0.0.1:${port}`;
    const env = {
        ...process.env,
        PORT: String(port),
        DESKTOP_APP_MODE: "true",
        FRONTEND_URL: "http://localhost:3000,null",
    };

    const backendPython = fs.existsSync(path.join("backend", ".venv", "bin", "python"))
        ? path.join(".venv", "bin", "python")
        : "python3";

    const sidecar = spawn(
        backendPython,
        ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", String(port)],
        {
            cwd: "backend",
            env,
            stdio: ["ignore", "pipe", "pipe"],
        }
    );
    let stderr = "";
    sidecar.stderr.on("data", (chunk) => {
        stderr += String(chunk);
    });

    try {
        const startedAt = Date.now();
        let healthy = false;
        while (Date.now() - startedAt < 120000) {
            try {
                await checkHealth(apiBase);
                healthy = true;
                break;
            } catch (_error) {
                await new Promise((resolve) => setTimeout(resolve, 500));
            }
        }
        if (!healthy) {
            throw new Error(`Sidecar health check timed out. ${stderr.trim()}`);
        }
        await run("node", ["-e", "process.exit(0)"]);
        process.stdout.write("Desktop smoke test passed.\n");
    } finally {
        sidecar.kill("SIGTERM");
    }
}

main().catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
});
