const SAFE_RELATIVE_PATH = /^\/[a-zA-Z0-9/_-]*$/;

export function sanitizeRedirectPath(next: string | null | undefined, fallback = "/dashboard"): string {
  if (!next || !SAFE_RELATIVE_PATH.test(next) || next.startsWith("//")) {
    return fallback;
  }
  return next;
}
