// 60 frames (1s) wasn't enough for the GPU to reach a steady clock state - run-to-run medians
// varied ~25%. 3s of warmup and 10s of measurement is far more repeatable.
const WARMUP_FRAMES = 180
const MEASURE_FRAMES = 600

export interface ScriptedInput {
  x: number
  y: number
  dx: number
  dy: number
}

/**
 * Reproducible timing runs.
 *
 * Input is a pure function of the frame index so every solver sees byte-identical strokes,
 * and timings are recorded GPU-side with no readback until the run ends - a live overlay
 * can't be compared between solvers because timings jitter and the readbacks themselves
 * cost something.
 */
export class Benchmark {
  private frame = 0
  private running = false

  get active(): boolean {
    return this.running
  }

  /** True once warm-up is over and frames are being recorded. */
  get measuring(): boolean {
    return this.running && this.frame >= WARMUP_FRAMES
  }

  /** Index into the recording, or -1 during warm-up. */
  get recordIndex(): number {
    return this.measuring ? this.frame - WARMUP_FRAMES : -1
  }

  get totalFrames(): number {
    return WARMUP_FRAMES + MEASURE_FRAMES
  }

  get measureFrames(): number {
    return MEASURE_FRAMES
  }

  get progress(): string {
    return this.frame < WARMUP_FRAMES
      ? `warmup ${this.frame}/${WARMUP_FRAMES}`
      : `measuring ${this.frame - WARMUP_FRAMES}/${MEASURE_FRAMES}`
  }

  start(): void {
    this.frame = 0
    this.running = true
  }

  stop(): void {
    this.running = false
  }

  /** Advances a frame. Returns true when the run has just finished. */
  advance(): boolean {
    if (!this.running) return false
    this.frame++
    if (this.frame < this.totalFrames) return false
    this.running = false
    return true
  }

  /**
   * A Lissajous sweep, so the stroke covers the domain and keeps generating fresh vortices
   * rather than settling into a steady state.
   */
  input(width: number, height: number): ScriptedInput {
    const t = this.frame * 0.02
    const position = (phase: number, freq: number, size: number) =>
      size * (0.5 + 0.32 * Math.sin(freq * t + phase))

    const x = position(0, 1.0, width)
    const y = position(Math.PI / 3, 1.37, height)
    const previousT = t - 0.02
    const px = width * (0.5 + 0.32 * Math.sin(1.0 * previousT))
    const py = height * (0.5 + 0.32 * Math.sin(1.37 * previousT + Math.PI / 3))

    return { x, y, dx: x - px, dy: y - py }
  }
}

export interface ScopeStats {
  name: string
  mean: number
  median: number
  p95: number
  min: number
}

export function summarise(timings: { name: string; samples: number[] }[]): ScopeStats[] {
  return timings
    .filter((scope) => scope.samples.length > 0)
    .map(({ name, samples }) => {
      const sorted = [...samples].sort((a, b) => a - b)
      const at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))]
      return {
        name,
        mean: samples.reduce((sum, v) => sum + v, 0) / samples.length,
        median: at(0.5),
        p95: at(0.95),
        min: sorted[0],
      }
    })
}

export function formatReport(label: string, stats: ScopeStats[], residual: number | null): string {
  // Median first: it reproduces to ~5% between runs, while the mean swings ~20% because p95
  // is 3-5x median (occasional multi-ms stalls, likely compositor contention).
  const width = Math.max(...stats.map((s) => s.name.length))
  const rows = stats.map(
    (s) =>
      `  ${s.name.padEnd(width)}  median ${s.median.toFixed(3)}  mean ${s.mean.toFixed(3)}` +
      `  p95 ${s.p95.toFixed(3)}  min ${s.min.toFixed(3)}`,
  )
  const residualLine = residual === null ? '' : `\n  residual ${residual.toExponential(3)}`
  return `benchmark [${label}]\n${rows.join('\n')}${residualLine}`
}
