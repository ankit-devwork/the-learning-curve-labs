import { BackendMeCard } from "@/components/auth/backend-me-card";

export function DevBackendMeCard() {
  if (process.env.NEXT_PUBLIC_SHOW_DEV_PANEL !== "true") {
    return null;
  }
  return <BackendMeCard />;
}
