-- Step 1.8: Excel analysis results (profile, charts, narrative summary)

alter table public.documents
  add column if not exists excel_profile jsonb,
  add column if not exists excel_charts jsonb,
  add column if not exists excel_summary text;
