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

function formatCompactValue(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}M`;
  }
  if (abs >= 10_000) {
    return `${(value / 1_000).toLocaleString(undefined, { maximumFractionDigits: 1 })}K`;
  }
  return formatValue(value);
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

function formatAxisDate(label: string): string {
  const parsed = Date.parse(label);
  if (!Number.isNaN(parsed)) {
    return new Date(parsed).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }
  return label.length > 10 ? `${label.slice(0, 9)}…` : label;
}

function tickTextAnchor(index: number, count: number): "start" | "middle" | "end" {
  if (index === 0) return "start";
  if (index === count - 1) return "end";
  return "middle";
}

function BarChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const max = Math.max(...values, 1);

  return (
    <div className="space-y-3">
      {labels.map((label, index) => (
        <div key={`${label}-${index}`} className="space-y-1.5">
          <div className="flex justify-between gap-3 text-xs">
            <span className="truncate text-muted-foreground">{formatLabel(label)}</span>
            <span className="shrink-0 font-medium tabular-nums">{formatValue(values[index])}</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-muted/80">
            <div
              className="h-full rounded-full"
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
  const width = 680;
  const height = 280;
  const padding = { top: 24, right: 36, bottom: 64, left: 56 };
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
  const tickIndexes = [0, Math.floor((labels.length - 1) / 2), labels.length - 1].filter(
    (value, index, array) => array.indexOf(value) === index && value >= 0,
  );

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full min-w-[340px]"
        role="img"
        aria-label="Line chart"
      >
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
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                className="fill-muted-foreground text-[11px]"
              >
                {formatCompactValue(tickValue)}
              </text>
            </g>
          );
        })}
        <line
          x1={padding.left}
          y1={padding.top + plotHeight}
          x2={padding.left + plotWidth}
          y2={padding.top + plotHeight}
          stroke="hsl(var(--border))"
        />
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
            <circle
              cx={point.x}
              cy={point.y}
              r={5}
              fill="hsl(var(--background))"
              stroke={CHART_COLORS[0]}
              strokeWidth={2}
            />
            <title>
              {formatLabel(point.label)}: {formatValue(point.value)}
            </title>
          </g>
        ))}
        {tickIndexes.map((dataIndex, tickPosition) => {
          const point = points[dataIndex];
          return (
            <text
              key={`tick-${dataIndex}`}
              x={point.x}
              y={height - 28}
              textAnchor={tickTextAnchor(tickPosition, tickIndexes.length)}
              className="fill-muted-foreground text-[11px]"
            >
              {formatAxisDate(point.label)}
            </text>
          );
        })}
      </svg>
    </div>
  );
}

function PieChartView({ labels, values }: { labels: string[]; values: number[] }) {
  const total = values.reduce((sum, value) => sum + value, 0) || 1;
  const size = 260;
  const center = size / 2;
  const radius = 96;
  const innerRadius = 58;

  let startAngle = -Math.PI / 2;
  const slices = values.map((value, index) => {
    const angle = (value / total) * Math.PI * 2;
    const endAngle = startAngle + angle;
    const path = describeArc(center, center, radius, innerRadius, startAngle, endAngle);
    const slice = {
      path,
      color: CHART_COLORS[index % CHART_COLORS.length],
      label: labels[index],
      value,
      percent: (value / total) * 100,
    };
    startAngle = endAngle;
    return slice;
  });

  return (
    <div className="flex flex-col items-center gap-8 lg:flex-row lg:items-center lg:justify-center">
      <div className="relative shrink-0">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="h-[240px] w-[240px]"
          role="img"
          aria-label="Pie chart"
          shapeRendering="geometricPrecision"
        >
          {slices.map((slice, index) => (
            <path
              key={`${slice.label}-${index}`}
              d={slice.path}
              fill={slice.color}
              stroke="hsl(var(--background))"
              strokeWidth={3}
              strokeLinejoin="round"
            >
              <title>
                {slice.label}: {formatValue(slice.value)} ({slice.percent.toFixed(1)}%)
              </title>
            </path>
          ))}
          <text
            x={center}
            y={center - 6}
            textAnchor="middle"
            className="fill-foreground text-[13px] font-semibold"
          >
            {formatCompactValue(total)}
          </text>
          <text
            x={center}
            y={center + 14}
            textAnchor="middle"
            className="fill-muted-foreground text-[11px]"
          >
            total
          </text>
        </svg>
      </div>
      <div className="grid w-full max-w-md gap-2.5 sm:grid-cols-2">
        {slices.map((slice, index) => (
          <div
            key={`legend-${slice.label}-${index}`}
            className="flex items-center gap-2.5 rounded-md border border-border/60 bg-muted/30 px-3 py-2 text-sm"
          >
            <span
              className="inline-block h-3.5 w-3.5 shrink-0 rounded-full"
              style={{ backgroundColor: slice.color }}
            />
            <span className="min-w-0 flex-1 truncate font-medium">{slice.label}</span>
            <span className="shrink-0 tabular-nums text-muted-foreground">
              {slice.percent.toFixed(1)}%
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
  const width = 680;
  const height = 320;
  const padding = { top: 28, right: 32, bottom: 56, left: 68 };
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

  const yTicks = [0, 0.5, 1];
  const xTicks = [0, 0.5, 1];

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full min-w-[340px]"
        role="img"
        aria-label="Scatter chart"
      >
        {yTicks.map((fraction) => {
          const y = padding.top + plotHeight * (1 - fraction);
          const tickValue = minY + yRange * fraction;
          return (
            <g key={`y-grid-${fraction}`}>
              <line
                x1={padding.left}
                y1={y}
                x2={padding.left + plotWidth}
                y2={y}
                stroke="hsl(var(--border))"
                strokeDasharray="4 4"
                opacity={fraction === 0 ? 1 : 0.7}
              />
              <text
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                className="fill-muted-foreground text-[11px]"
              >
                {formatCompactValue(tickValue)}
              </text>
            </g>
          );
        })}
        {xTicks.map((fraction) => {
          const x = padding.left + plotWidth * fraction;
          const tickValue = minX + xRange * fraction;
          return (
            <g key={`x-grid-${fraction}`}>
              <line
                x1={x}
                y1={padding.top}
                x2={x}
                y2={padding.top + plotHeight}
                stroke="hsl(var(--border))"
                strokeDasharray="4 4"
                opacity={0.5}
              />
              <text
                x={x}
                y={height - 32}
                textAnchor={tickTextAnchor(xTicks.indexOf(fraction), xTicks.length)}
                className="fill-muted-foreground text-[11px]"
              >
                {formatCompactValue(tickValue)}
              </text>
            </g>
          );
        })}
        <rect
          x={padding.left}
          y={padding.top}
          width={plotWidth}
          height={plotHeight}
          fill="hsl(var(--muted) / 0.25)"
          stroke="hsl(var(--border))"
          rx={4}
        />
        {projected.map((point, index) => (
          <circle
            key={`${point.label}-${index}`}
            cx={point.px}
            cy={point.py}
            r={4}
            fill={CHART_COLORS[0]}
            fillOpacity={0.72}
            stroke="hsl(var(--background))"
            strokeWidth={1.5}
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
