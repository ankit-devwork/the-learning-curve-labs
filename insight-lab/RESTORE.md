# Restore scaffold (empty GitHub repo)

The initial commit was created in Cursor Cloud but could not be pushed automatically
(`cursor[bot]` lacks write access to `ankit-devwork/insight-lab`).

## Option A — PowerShell restore script (Windows)

From your **empty** `insight-lab` clone:

1. Copy `scripts/restore-insight-lab.ps1` from this folder into your clone (or re-clone after push).
2. Run:

```powershell
cd D:\Mine\Learining\GenAI\python\insight-lab
powershell -ExecutionPolicy Bypass -File .\scripts\restore-insight-lab.ps1
git push -u origin main
```

## Option B — Git bundle (any OS)

If you have `insight-lab.bundle` (generated in Cursor workspace):

```bash
cd insight-lab
git pull /path/to/insight-lab.bundle main
git push -u origin main
```

Generate bundle in workspace:

```bash
cd insight-lab && git bundle create insight-lab.bundle main
```

## Option C — Push yourself from Cursor Cloud

Open the Cursor Cloud workspace, `cd insight-lab`, and push with **your** GitHub credentials:

```bash
git remote set-url origin https://github.com/ankit-devwork/insight-lab.git
git push -u origin main
```

## After push

Re-clone or `git pull` — the repo will no longer be empty.

```bash
git clone https://github.com/ankit-devwork/insight-lab.git
```
