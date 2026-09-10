export const USERNAME_INPUT_MAX_LENGTH = 128;
export const PASSWORD_INPUT_MAX_LENGTH = 256;

type AuthValidationResult = {
  username: string;
  error: string | null;
};

export function validateAuthCredentials(
  username: string,
  password: string,
  passwordConfirmation?: string,
): AuthValidationResult {
  const normalizedUsername = username.trim().normalize("NFKC");
  const usernameLength = Array.from(normalizedUsername).length;
  if (usernameLength < 3 || usernameLength > 64) {
    return { username: normalizedUsername, error: "用户名需包含 3 至 64 个字符。" };
  }
  if (!/^[\p{L}\p{N}][\p{L}\p{N}._-]*$/u.test(normalizedUsername)) {
    return {
      username: normalizedUsername,
      error: "用户名需以文字或数字开头，仅可包含文字、数字、点、下划线和连字符。",
    };
  }

  // Match Python's code-point limits; HTML maxlength counts UTF-16 code units.
  const passwordLength = Array.from(password).length;
  if (passwordLength < 8 || passwordLength > 128) {
    return { username: normalizedUsername, error: "密码需包含 8 至 128 个字符。" };
  }
  if (passwordConfirmation !== undefined && password !== passwordConfirmation) {
    return { username: normalizedUsername, error: "两次输入的密码不一致。" };
  }
  return { username: normalizedUsername, error: null };
}
