// Validate local generated fixtures with the official, exactly pinned library.
// Usage: node validate_glb.cjs NPM_PREFIX GLB_DIRECTORY
// The validator and generated files stay local; this helper uploads nothing.
"use strict";

const fs = require("node:fs");
const path = require("node:path");

async function main() {
  if (process.argv.length !== 4) {
    throw new Error("Usage: node validate_glb.cjs NPM_PREFIX GLB_DIRECTORY");
  }
  const prefix = path.resolve(process.argv[2]);
  const directory = path.resolve(process.argv[3]);
  const packagePath = path.join(prefix, "node_modules", "gltf-validator");
  const installed = require(path.join(packagePath, "package.json"));
  if (installed.version !== "2.0.0-dev.3.10") {
    throw new Error(`Expected gltf-validator@2.0.0-dev.3.10, got ${installed.version}`);
  }
  const validator = require(packagePath);
  const files = fs.readdirSync(directory).filter((name) => name.endsWith(".glb")).sort();
  if (files.length === 0) {
    throw new Error("No .glb files found; validation did not run");
  }
  let failed = false;
  for (const filename of files) {
    const data = fs.readFileSync(path.join(directory, filename));
    const report = await validator.validateBytes(new Uint8Array(data), {
      uri: filename,
      format: "glb",
      writeTimestamp: false,
      maxIssues: 0,
      externalResourceFunction: (uri) => Promise.reject(
        new Error(`Self-contained GLB must not request an external resource: ${uri}`)
      ),
    });
    const issues = report.issues;
    const codes = {};
    for (const issue of issues.messages) {
      codes[issue.code] = (codes[issue.code] || 0) + 1;
    }
    console.log(JSON.stringify({
      file: filename,
      validator: validator.version(),
      errors: issues.numErrors,
      warnings: issues.numWarnings,
      infos: issues.numInfos,
      hints: issues.numHints,
      codes,
    }));
    failed ||= issues.numErrors > 0 || issues.numWarnings > 0;
  }
  if (failed) {
    throw new Error("Khronos GLB validation reported errors or warnings");
  }
}

main().catch((error) => {
  console.error(error.message || String(error));
  process.exitCode = 1;
});
