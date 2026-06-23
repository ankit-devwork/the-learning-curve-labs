import { BrandMark } from "@/components/layout/brand-mark";

type AuthLayoutProps = {
  children: React.ReactNode;
};

export function AuthLayout({ children }: AuthLayoutProps) {
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-2">
      <aside className="relative hidden overflow-hidden bg-primary px-10 py-12 text-primary-foreground lg:flex lg:flex-col lg:justify-between">
        <BrandMark inverted href={null} />
        <div className="space-y-4">
          <h1 className="text-3xl font-semibold leading-tight tracking-tight">
            Turn documents and spreadsheets into insights you can use.
          </h1>
          <p className="max-w-md text-sm leading-relaxed text-primary-foreground/85">
            Upload course materials or data files, ask questions in plain English, and check
            understanding with AI-generated quizzes.
          </p>
          <ul className="space-y-2 text-sm text-primary-foreground/80">
            <li>· Summaries and cited answers from PDFs and Word docs</li>
            <li>· Charts and Q&A for Excel and CSV files</li>
            <li>· Quizzes with topic progress tracking</li>
          </ul>
        </div>
        <p className="text-xs text-primary-foreground/60">InsightLab pilot</p>
      </aside>
      <div className="flex items-center justify-center bg-[hsl(var(--shell))] p-6">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  );
}
