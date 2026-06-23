export type WorkspaceAccessRole = "owner" | "editor" | "viewer";

export function canEditWorkspace(role?: WorkspaceAccessRole | null): boolean {
  return role === "owner" || role === "editor";
}

export function workspaceRoleLabel(role?: WorkspaceAccessRole | null): string {
  switch (role) {
    case "owner":
      return "Owner";
    case "editor":
      return "Editor";
    case "viewer":
      return "Viewer";
    default:
      return "Member";
  }
}

export function cacheResponseLabel(cached?: boolean, cacheMatch?: string, similarity?: number): string | null {
  if (!cached) {
    return null;
  }
  if (cacheMatch === "semantic" && similarity != null) {
    return `Similar question cached (${Math.round(similarity * 100)}% match)`;
  }
  return "Cached response";
}
