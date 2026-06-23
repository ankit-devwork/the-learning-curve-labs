import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function FileListSkeleton() {
  return (
    <ul className="divide-y overflow-hidden rounded-xl border bg-card shadow-sm" aria-hidden>
      {Array.from({ length: 4 }).map((_, index) => (
        <li key={index} className="flex items-center justify-between gap-3 px-4 py-3.5">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <Skeleton className="h-10 w-10 shrink-0 rounded-lg" />
            <div className="min-w-0 flex-1 space-y-2">
              <Skeleton className="h-4 w-3/5 max-w-[240px]" />
              <Skeleton className="h-3 w-2/5 max-w-[160px]" />
            </div>
          </div>
          <Skeleton className="h-6 w-16 rounded-full" />
        </li>
      ))}
    </ul>
  );
}

export function DocumentDetailSkeleton() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading document">
      <div className="space-y-3 border-b pb-6">
        <Skeleton className="h-4 w-36" />
        <Skeleton className="h-8 w-2/3 max-w-md" />
        <div className="flex gap-2">
          <Skeleton className="h-6 w-20 rounded-md" />
          <Skeleton className="h-6 w-16 rounded-md" />
        </div>
      </div>

      <Card className="shadow-sm">
        <CardHeader>
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-4 w-full max-w-sm" />
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-4/5" />
          <Skeleton className="h-4 w-3/5" />
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="shadow-sm">
          <CardHeader>
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-4 w-full max-w-xs" />
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-10 w-full rounded-md" />
            <Skeleton className="h-24 w-full rounded-md" />
          </CardContent>
        </Card>
        <Card className="shadow-sm">
          <CardHeader>
            <Skeleton className="h-5 w-36" />
            <Skeleton className="h-4 w-full max-w-sm" />
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-10 w-full rounded-md" />
            <Skeleton className="h-20 w-full rounded-md" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export function ExcelDetailSkeleton() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading spreadsheet">
      <div className="space-y-3 border-b pb-6">
        <Skeleton className="h-4 w-36" />
        <Skeleton className="h-8 w-2/3 max-w-md" />
        <Skeleton className="h-6 w-24 rounded-md" />
      </div>

      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-4 w-48" />
        </CardHeader>
        <CardContent className="space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-full max-w-md" />
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-10 w-full rounded-md" />
          <Skeleton className="h-16 w-full rounded-md" />
        </CardContent>
      </Card>

      <div className="space-y-4">
        <Skeleton className="h-6 w-40" />
        <Card>
          <CardHeader>
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-4 w-56" />
          </CardHeader>
          <CardContent>
            <Skeleton className="h-64 w-full rounded-lg" />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export function ProcessingContentSkeleton({ lines = 5 }: { lines?: number }) {
  const widths = ["w-full", "w-full", "w-11/12", "w-4/5", "w-3/5", "w-2/3"];
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: lines }).map((_, index) => (
        <Skeleton key={index} className={cn("h-4", widths[index % widths.length])} />
      ))}
    </div>
  );
}

export function ChartBuilderSkeleton() {
  return (
    <Card>
      <CardHeader>
        <Skeleton className="h-5 w-36" />
        <Skeleton className="h-4 w-64" />
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-3">
          <Skeleton className="h-10 w-full rounded-md" />
          <Skeleton className="h-10 w-full rounded-md" />
          <Skeleton className="h-10 w-full rounded-md" />
        </div>
        <Skeleton className="h-10 w-32 rounded-md" />
      </CardContent>
    </Card>
  );
}
