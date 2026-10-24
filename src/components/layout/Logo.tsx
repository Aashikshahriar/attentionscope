export function Logo({ className }: { className?: string }) {
  // A 4×4 causal attention pattern (lower triangle) inside a lens.
  const cells = [
    [0, 0, 1],
    [1, 0, 0.55],
    [1, 1, 0.75],
    [2, 0, 0.3],
    [2, 1, 0.45],
    [2, 2, 0.9],
    [3, 0, 0.25],
    [3, 1, 0.35],
    [3, 2, 0.6],
    [3, 3, 0.8],
  ] as const;
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <defs>
        <linearGradient id="as-lg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#8B8CFF" />
          <stop offset="1" stopColor="#38D9C5" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#111827" stroke="url(#as-lg)" strokeWidth="1.5" />
      {cells.map(([r, c, o]) => (
        <rect
          key={`${r}${c}`}
          x={7 + c * 4.6}
          y={7 + r * 4.6}
          width="3.8"
          height="3.8"
          rx="0.8"
          fill="url(#as-lg)"
          opacity={o}
        />
      ))}
    </svg>
  );
}
