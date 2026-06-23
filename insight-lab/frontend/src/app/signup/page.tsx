import { SignUpForm } from "@/components/auth/signup-form";
import { AuthLayout } from "@/components/layout/auth-layout";
import { BrandMark } from "@/components/layout/brand-mark";

export default function SignUpPage() {
  return (
    <AuthLayout>
      <div className="mb-8 lg:hidden">
        <BrandMark href={null} />
      </div>
      <SignUpForm />
    </AuthLayout>
  );
}
