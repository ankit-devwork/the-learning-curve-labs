import { LoginForm } from "@/components/auth/login-form";
import { AuthLayout } from "@/components/layout/auth-layout";
import { BrandMark } from "@/components/layout/brand-mark";

export default function LoginPage() {
  return (
    <AuthLayout>
      <div className="mb-8 lg:hidden">
        <BrandMark href={null} />
      </div>
      <LoginForm />
    </AuthLayout>
  );
}
