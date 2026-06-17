"use client";

import type { ExcelChart } from "@/lib/api";

const CHART_COLORS = [
  "hsl(217, 91%, 60%)",
  "hsl(142, 71%, 45%)",
  "hsl(38, 92%, 50%)",
  "hsl(280, 67%, 58%)",
  "hsl(0, 84%, 60%)",
  "hsl(189, 94%, 43%)",
  "hsl(24, 95%, 53%)",
];

function formatValue(value: number): string {
  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function formatLabel(label: string): string {
  const parsed = Date.parse(label);
  if (!Number.isNaN(parsed)) {
    return new Date(parsed).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  }
  return label;
}

function BarChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const max = Math.max(...values, 1);

  return (
    <div className="space-y-3">
      {labels.map((label, index) => (
        <div key={`${label}-${index}`} className="space-y-1">
          <div className="flex justify-between gap-3 text-xs">
            <span className="truncate text-muted-foreground">{formatLabel(label)}</span>
            <span className="shrink-0 font-medium tabular-nums">{formatValue(values[index])}</span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${(values[index] / max) * 100}%`,
                backgroundColor: CHART_COLORS[index % CHART_COLORS.length],
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function LineChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const width = 640;
  const height = 240;
  const padding = { top: 16, right: 16, bottom: 48, left: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const yRange = maxY - minY || 1;

  const points = values.map((value, index) => {
    const x =
      values.length === 1
        ? padding.left + plotWidth / 2
        : padding.left + (index / (values.length - 1)) * plotWidth;
    const y = padding.top + plotHeight - ((value - minY) / yRange) * plotHeight;
    return { x, y, value, label: labels[index] };
  });

  const polyline = points.map((point) => `${point.x},${point.y}`).join(" ");
  const tickIndexes = new Set([0, Math.floor((labels.length - 1) / 2), labels.length - 1].filter((i) => i >= 0));

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full min-w-[320px]" role="img">
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = padding.top + plotHeight * (1 - fraction);
          const tickValue = minY + yRange * fraction;
          return (
            <g key={fraction}>
              <line
                x1={padding.left}
                y1={y}
                x2={padding.left + plotWidth}
                y2={y}
                stroke="hsl(var(--border))"
                strokeDasharray="4 4"
              />
              <text
                x={padding.left - 8}
                y={y + 4}
                textAnchor="end"
                className="fill-muted-foreground text-[10px]"
              >
                {formatValue(tickValue)}
              </text>
            </g>
          );
        })}
        <polyline
          fill="none"
          stroke={CHART_COLORS[0]}
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
          points={polyline}
        />
        {points.map((point, index) => (
          <g key={`${point.label}-${index}`}>
            <circle cx={point.x} cy={point.y} r={4} fill={CHART_COLORS[0]} />
            <title>
              {formatLabel(point.label)}: {formatValue(point.value)}
            </title>
          </g>
        ))}
        {points.map((point, index) =>
          tickIndexes.has(index) ? (
            <text
              key={`tick-${index}`}
              x={point.x}
              y={height - 8}
              textAnchor="middle"
              className="fill-muted-foreground text-[10px]"
            >
              {formatLabel(point.label)}
            </text>
          ) : null,
        )}
      </svg>
    </div>
  );
}

function PieChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const total = values.reduce((sum, value) => sum + value, 0) || 1;
  const size = 220;
  const center = size / 2;
  const radius = 88;
  const innerRadius = 48;

  let startAngle = -Math.PI / 2;
  const slices = values.map((value, index) => {
    const angle = (value / total) * Math.PI * 2;
    const endAngle = startAngle + angle;
    const path = describeArc(center, center, radius, innerRadius, startAngle, endAngle);
    const midAngle = startAngle + angle / 2;
    const slice = {
      path,
      color: CHART_COLORS[index % CHART_COLORS.length],
      label: labels[index],
      value,
      percent: (value / total) * 100,
      midAngle,
    };
    startAngle = endAngle;
    return slice;
  });

  return (
    <div className="flex flex-col items-center gap-6 md:flex-row md:items-start">
      <svg viewBox={`0 0 ${size} ${size}`} className="h-[220px] w-[220px] shrink-0" role="img">
        {slices.map((slice, index) => (
          <path key={`${slice.label}-${index}`} d={slice.path} fill={slice.color}>
            <title>
              {slice.label}: {formatValue(slice.value)} ({slice.percent.toFixed(1)}%)
            </title>
          </path>
        ))}
      </svg>
      <div className="grid w-full gap-2 sm:grid-cols-2">
        {slices.map((slice, index) => (
          <div key={`legend-${slice.label}-${index}`} className="flex items-center gap-2 text-sm">
            <span
              className="inline-block h-3 w-3 shrink-0 rounded-sm"
              style={{ backgroundColor: slice.color }}
            />
            <span className="truncate">{slice.label}</span>
            <span className="ml-auto shrink-0 tabular-nums text-muted-foreground">
              {formatValue(slice.value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function describeArc(
  cx: number,
  cy: number,
  outerRadius: number,
  innerRadius: number,
  startAngle: number,
  endAngle: number,
): string {
  const startOuter = polarToCartesian(cx, cy, outerRadius, endAngle);
  const endOuter = polarToCartesian(cx, cy, outerRadius, startAngle);
  const startInner = polarToCartesian(cx, cy, innerRadius, endAngle);
  const endInner = polarToCartesian(cx, cy, innerRadius, startAngle);
  const largeArc = endAngle - startAngle <= Math.PI ? 0 : 1;

  return [
    `M ${startOuter.x} ${startOuter.y}`,
    `A ${outerRadius} ${outerRadius} 0 ${largeArc} 0 ${endOuter.x} ${endOuter.y}`,
    `L ${endInner.x} ${endInner.y}`,
    `A ${innerRadius} ${innerRadius} 0 ${largeArc} 1 ${startInner.x} ${startInner.y}`,
    "Z",
  ].join(" ");
}

function polarToCartesian(cx: number, cy: number, radius: number, angle: number) {
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  };
}

function ScatterChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const width = 640;
  const height = 280;
  const padding = { top: 16, right: 16, bottom: 40, left: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const points = labels
    .map((label, index) => ({
      x: Number.parseFloat(label),
      y: values[index],
      label,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (points.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Scatter preview unavailable — x values must be numeric.
      </p>
    );
  }

  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));
  const xRange = maxX - minX || 1;
  const yRange = maxY - minY || 1;

  const projected = points.map((point) => ({
    ...point,
    px: padding.left + ((point.x - minX) / xRange) * plotWidth,
    py: padding.top + plotHeight - ((point.y - minY) / yRange) * plotHeight,
  }));

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full min-w-[320px]" role="img">
        {[0, 0.5, 1].map((fraction) => {
          const y = padding.top + plotHeight * (1 - fraction);
          const tickValue = minY + yRange * fraction;
          return (
            <text
              key={`y-${fraction}`}
              x={padding.left - 8}
              y={y + 4}
              textAnchor="end"
              className="fill-muted-foreground text-[10px]"
            >
              {formatValue(tickValue)}
            </text>
          );
        })}
        {[0, 0.5, 1].map((fraction) => {
          const x = padding.left + plotWidth * fraction;
          const tickValue = minX + xRange * fraction;
          return (
            <text
              key={`x-${fraction}`}
              x={x}
              y={height - 10}
              textAnchor="middle"
              className="fill-muted-foreground text-[10px]"
            >
              {formatValue(tickValue)}
            </text>
          );
        })}
        <rect
          x={padding.left}
          y={padding.top}
          width={plotWidth}
          height={plotHeight}
          fill="none"
          stroke="hsl(var(--border))"
        />
        {projected.map((point, index) => (
          <circle
            key={`${point.label}-${index}`}
            cx={point.px}
            cy={point.py}
            r={3.5}
            fill={CHART_COLORS[index % CHART_COLORS.length]}
            opacity={0.85}
          >
            <title>
              ({formatValue(point.x)}, {formatValue(point.y)})
            </title>
          </circle>
        ))}
      </svg>
    </div>
  );
}

export function ExcelChartView({ chart }: { chart: ExcelChart }) {
  switch (chart.chart_type) {
    case "line":
      return <LineChartView labels={chart.labels} values={chart.values} />;
    case "pie":
      return <PieChartView labels={chart.labels} values={chart.values} />;
    case "scatter":
      return <ScatterChartView labels={chart.labels} values={chart.values} />;
    case "bar":
    default:
      return <BarChartView labels={chart.labels} values={chart.values} />;
  }
}
