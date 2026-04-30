const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const repoRoot = path.join(__dirname, "..", "..");
const backendVenvPython = path.join(repoRoot, "backend", ".venv", "bin", "python");
const pythonExecutable = fs.existsSync(backendVenvPython)
    ? backendVenvPython
    : "python3";

const args = [
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--name",
    "nesy-backend",
    "--distpath",
    path.join(repoRoot, "desktop", "sidecar"),
    path.join(repoRoot, "backend", "desktop_sidecar.py"),
];

const result = spawnSync(pythonExecutable, args, {
    stdio: "inherit",
    cwd: repoRoot,
    env: process.env,
});

if (result.status !== 0) {
    const sidecarDir = path.join(repoRoot, "desktop", "sidecar");
    fs.mkdirSync(sidecarDir, { recursive: true });
    const fallbackLauncher = path.join(sidecarDir, "nesy-backend");
    fs.writeFileSync(
        fallbackLauncher,
        "#!/usr/bin/env bash\nPORT=8000\nif [ \"$1\" = \"--port\" ] && [ -n \"$2\" ]; then PORT=\"$2\"; fi\nSCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\ncd \"$SCRIPT_DIR/../backend\" || exit 1\npython3 -m uvicorn main:app --host 127.0.0.1 --port \"$PORT\"\n",
        { encoding: "utf-8" }
    );
    fs.chmodSync(fallbackLauncher, 0o755);
    process.stderr.write(
        "PyInstaller build failed; created python3 fallback launcher at desktop/sidecar/nesy-backend\n"
    );
}
