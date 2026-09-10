import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/auth-validation.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const {
  PASSWORD_INPUT_MAX_LENGTH,
  USERNAME_INPUT_MAX_LENGTH,
  validateAuthCredentials,
} = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

test("username normalization matches the account key without changing password", () => {
  assert.deepEqual(validateAuthCredentials("  \uff21\uff22\uff23_123  ", "password42"), {
    username: "ABC_123",
    error: null,
  });
  assert.notEqual(validateAuthCredentials("ABC_123", "password42", "PASSWORD42").error, null);
});

test("username allows normalized letters and numbers but rejects unsafe characters", () => {
  for (const username of [".invalid", "two words", "a@b", "a/b", "ab"]) {
    assert.notEqual(validateAuthCredentials(username, "password42").error, null, username);
  }
  for (const username of ["medical_user", "user-01", "abc.def", "\u7528\u6237\u7532"]) {
    assert.equal(validateAuthCredentials(username, "password42").error, null, username);
  }
});

test("password bounds count Unicode code points rather than UTF-16 code units", () => {
  const supplementaryCharacter = "\u{1F512}";
  assert.notEqual(validateAuthCredentials("valid_user", supplementaryCharacter.repeat(4)).error, null);
  assert.equal(validateAuthCredentials("valid_user", supplementaryCharacter.repeat(8)).error, null);
  assert.equal(validateAuthCredentials("valid_user", supplementaryCharacter.repeat(128)).error, null);
  assert.notEqual(validateAuthCredentials("valid_user", supplementaryCharacter.repeat(129)).error, null);
  assert.equal(validateAuthCredentials("valid_user", "a".repeat(128)).error, null);
  assert.notEqual(validateAuthCredentials("valid_user", "a".repeat(129)).error, null);
});

test("input bounds allow the largest valid supplementary-plane credentials", () => {
  const username = "\u{20000}".repeat(64);
  const password = "\u{1F512}".repeat(128);
  assert.equal(validateAuthCredentials(username, password, password).error, null);
  assert.equal(username.length, USERNAME_INPUT_MAX_LENGTH);
  assert.equal(password.length, PASSWORD_INPUT_MAX_LENGTH);
  assert.notEqual(validateAuthCredentials(`${username}\u{20000}`, password).error, null);
});

test("confirmation is required for registration but not for login", () => {
  assert.equal(validateAuthCredentials("valid_user", "password42").error, null);
  assert.notEqual(validateAuthCredentials("valid_user", "password42", "").error, null);
  assert.equal(validateAuthCredentials("valid_user", "password42", "password42").error, null);
});
