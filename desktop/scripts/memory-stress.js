const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const MINIMAL_PDF = `%PDF-1.1
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 18 Tf 20 120 Td (NeSy Memory Test) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000010 00000 n
0000000060 00000 n
0000000117 00000 n
0000000249 00000 n
0000000342 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
417
%%EOF
`;

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Request failed (${response.status}) for ${url}`);
    }
    return response.json();
}

async function postUpload(apiBase, pdfPaths) {
    const formData = new FormData();
    pdfPaths.forEach((pdfPath) => {
        const blob = new Blob([fs.readFileSync(pdfPath)], {
            type: "application/pdf",
        });
        formData.append("files", blob, path.basename(pdfPath));
    });
    const response = await fetch(`${apiBase}/api/graph/upload`, {
        method: "POST",
        body: formData,
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Upload failed (${response.status}): ${text}`);
    }
    return response.json();
}

async function waitForJob(apiBase, jobId) {
    for (let attempts = 0; attempts < 600; attempts += 1) {
        const payload = await getJson(`${apiBase}/api/jobs/${jobId}`);
        if (payload.status === "done") return payload;
        if (payload.status === "error") {
            throw new Error(payload.error || "Upload job failed");
        }
        await sleep(1000);
    }
    throw new Error(`Timed out waiting for job ${jobId}`);
}

async function main() {
    const port = Number(process.env.MEMORY_STRESS_PORT || 8766);
    const uploads = Number(process.env.MEMORY_STRESS_UPLOADS || 3);
    const filesPerUpload = Number(process.env.MEMORY_STRESS_FILES || 3);
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
    const tmpDir = fs.mkdtempSync(path.join(process.cwd(), "memory-stress-"));
    const pdfPaths = [];
    for (let idx = 0; idx < filesPerUpload; idx += 1) {
        const pdfPath = path.join(tmpDir, `sample-${idx + 1}.pdf`);
        fs.writeFileSync(pdfPath, MINIMAL_PDF, "utf-8");
        pdfPaths.push(pdfPath);
    }

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
        while (Date.now() - startedAt < 120000) {
            try {
                await getJson(`${apiBase}/health`);
                break;
            } catch (_error) {
                await sleep(500);
            }
        }

        const baseline = await getJson(`${apiBase}/api/runtime/memory`);
        let peak = baseline.rss_mb || 0;
        const diagnostics = await getJson(`${apiBase}/api/runtime/diagnostics`);
        if (!diagnostics.openai_configured) {
            process.stdout.write(
                `Memory stress skipped upload loop because OPENAI_API_KEY is not configured. baseline_mb=${baseline.rss_mb?.toFixed(2)}\n`
            );
            return;
        }
        for (let run = 0; run < uploads; run += 1) {
            const queued = await postUpload(apiBase, pdfPaths);
            await waitForJob(apiBase, queued.job_id);
            const sample = await getJson(`${apiBase}/api/runtime/memory`);
            peak = Math.max(peak, sample.rss_mb || 0);
        }
        const finalSample = await getJson(`${apiBase}/api/runtime/memory`);
        process.stdout.write(
            `Memory stress complete | baseline_mb=${baseline.rss_mb?.toFixed(2)} | peak_mb=${peak.toFixed(2)} | final_mb=${finalSample.rss_mb?.toFixed(2)}\n`
        );
    } catch (error) {
        throw new Error(`${error.message}\n${stderr}`);
    } finally {
        sidecar.kill("SIGTERM");
        fs.rmSync(tmpDir, { recursive: true, force: true });
    }
}

main().catch((error) => {
    process.stderr.write(`${error.message}\n`);
    process.exit(1);
});
