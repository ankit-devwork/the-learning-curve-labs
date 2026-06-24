import { apiFetch, type ExcelChart } from "@/lib/api";

function escapeCsvValue(value: string | number): string {
  const text = String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function downloadChartCsv(chart: ExcelChart, filename?: string): void {
  const lines = [
    "label,value",
    ...chart.labels.map((label, index) =>
      `${escapeCsvValue(label)},${escapeCsvValue(chart.values[index] ?? 0)}`,
    ),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${filename ?? chart.title.replace(/\s+/g, "-").toLowerCase()}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function drawBarChartPng(chart: ExcelChart): HTMLCanvasElement {
  const width = 900;
  const height = 520;
  const padding = { top: 48, right: 32, bottom: 72, left: 48 };
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Canvas not supported");
  }

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#111827";
  ctx.font = "bold 20px sans-serif";
  ctx.fillText(chart.title, padding.left, 28);

  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const max = Math.max(...chart.values, 1);
  const barWidth = plotWidth / Math.max(chart.labels.length, 1) - 12;
  const colors = ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ef4444", "#06b6d4"];

  chart.labels.forEach((label, index) => {
    const value = chart.values[index] ?? 0;
    const barHeight = (value / max) * plotHeight;
    const x = padding.left + index * (barWidth + 12);
    const y = padding.top + plotHeight - barHeight;
    ctx.fillStyle = colors[index % colors.length];
    ctx.fillRect(x, y, barWidth, barHeight);
    ctx.fillStyle = "#6b7280";
    ctx.font = "12px sans-serif";
    ctx.save();
    ctx.translate(x + barWidth / 2, height - 24);
    ctx.rotate(-0.4);
    ctx.textAlign = "right";
    ctx.fillText(label.slice(0, 18), 0, 0);
    ctx.restore();
  });

  return canvas;
}

export async function downloadChartPng(
  chart: ExcelChart,
  container: HTMLElement | null,
  filename?: string,
): Promise<void> {
  let canvas: HTMLCanvasElement | null = null;
  const svg = container?.querySelector("svg");
  if (svg) {
    const serializer = new XMLSerializer();
    const source = serializer.serializeToString(svg);
    const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const image = new Image();
    canvas = await new Promise<HTMLCanvasElement>((resolve, reject) => {
      image.onload = () => {
        const next = document.createElement("canvas");
        next.width = image.width || 900;
        next.height = image.height || 520;
        const ctx = next.getContext("2d");
        if (!ctx) {
          reject(new Error("Canvas not supported"));
          return;
        }
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, next.width, next.height);
        ctx.drawImage(image, 0, 0);
        URL.revokeObjectURL(url);
        resolve(next);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Failed to render chart"));
      };
      image.src = url;
    });
  } else {
    canvas = drawBarChartPng(chart);
  }

  const pngUrl = canvas.toDataURL("image/png");
  const anchor = document.createElement("a");
  anchor.href = pngUrl;
  anchor.download = `${filename ?? chart.title.replace(/\s+/g, "-").toLowerCase()}.png`;
  anchor.click();
}

export function downloadAnkiCsv(cards: Array<{ front: string; back: string }>, filename: string): void {
  const lines = ["Front,Back,Tags", ...cards.map((card) => {
    const front = card.front.replace(/"/g, '""');
    const back = card.back.replace(/"/g, '""');
    return `"${front}","${back}","InsightLab"`;
  })];
  downloadTextFile(lines.join("\n"), filename);
}

export function downloadTextFile(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function downloadAuthenticatedText(
  path: string,
  accessToken: string,
  filename: string,
): Promise<void> {
  const response = await apiFetch(path, accessToken);
  if (!response.ok) {
    throw new Error(`Download failed (${response.status})`);
  }
  downloadTextFile(await response.text(), filename);
}

export function downloadStudyGuidePdf(title: string, content: {
  overview?: string;
  key_terms?: Array<{ term: string; definition: string }>;
  sections?: Array<{ heading: string; bullets: string[] }>;
  sample_questions?: string[];
}): void {
  const html = `
    <!DOCTYPE html><html><head><title>${title}</title>
    <style>body{font-family:system-ui,sans-serif;padding:2rem;line-height:1.5}h1,h2{margin-top:1.5rem}</style>
    </head><body>
    <h1>${title}</h1>
    ${content.overview ? `<h2>Overview</h2><p>${content.overview}</p>` : ""}
    ${content.key_terms?.length ? `<h2>Key terms</h2><ul>${content.key_terms.map((t) => `<li><strong>${t.term}</strong>: ${t.definition}</li>`).join("")}</ul>` : ""}
    ${content.sections?.map((s) => `<h2>${s.heading}</h2><ul>${s.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>`).join("") ?? ""}
    ${content.sample_questions?.length ? `<h2>Sample questions</h2><ul>${content.sample_questions.map((q) => `<li>${q}</li>`).join("")}</ul>` : ""}
    </body></html>`;
  const win = window.open("", "_blank");
  if (!win) {
    return;
  }
  win.document.write(html);
  win.document.close();
  win.focus();
  win.print();
}
