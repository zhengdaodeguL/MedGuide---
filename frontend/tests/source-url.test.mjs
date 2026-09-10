import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/source-url.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { safeCitationSourceUrl } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`
);

test("citation links allow public HTTPS sources", () => {
  assert.equal(
    safeCitationSourceUrl("https://www.nhs.uk/conditions/common-cold/?view=full"),
    "https://www.nhs.uk/conditions/common-cold/?view=full",
  );
});

test("citation links reject credentials, private targets, and sensitive queries", () => {
  for (const value of [
    "http://example.org/reference",
    "https://localhost/reference",
    "https://127.0.0.1/reference",
    "https://10.0.0.1/reference",
    "https://user:password@example.org/reference",
    "https://example.org/reference#fragment",
    "https://evil.example/phish",
    "https://www.nhs.uk/reference?token=secret",
    "https://www.nhs.uk/reference?api%255fkey=secret",
    "https://www.nhs.uk/reference?AWSAccessKeyId=secret",
    "https://www.nhs.uk/reference?subscription-key=secret",
    "https://www.nhs.uk/reference?private_key=secret",
    "https://www.nhs.uk/reference?redirect=https%3A%2F%2Fother.test%2Fx%3Ftoken%3Dsecret",
    "https://www.nhs.uk/reference?redirect=https%253A%252F%252Fother.test%252Fx%253Fprivate_key%253Dsecret",
  ]) {
    assert.equal(safeCitationSourceUrl(value), null, value);
  }
});
