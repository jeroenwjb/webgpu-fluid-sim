const SAMPLE_WINDOW = 60
const UPDATE_INTERVAL_MS = 250

export interface ScopeTiming {
  name: string
  ms: number
}

/** Fastest frames in the window approximate the vsync period; p10 ignores early-delivery jitter. */
function refreshPeriodMs(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length * 0.1)]
}

/**
 * Frame time plus per-stage GPU timings. Frame time under rAF only tells you what the display
 * is doing, so it's reported alongside a limiter verdict; the GPU numbers are the real cost.
 */
export class Stats {
  private element: HTMLElement
  private samples: number[] = []
  private lastFrameTime = performance.now()
  private lastUpdate = 0
  private extraLine = ''
  private residual: number | null = null
  private controls: string[] = []

  constructor(element: HTMLElement) {
    this.element = element
  }

  /** Extra line under the timings, e.g. sim resolution. */
  setDetail(text: string): void {
    this.extraLine = text
  }

  /** Key bindings as [key, description] pairs; main.ts owns them. */
  setControls(bindings: [string, string][]): void {
    const keyWidth = Math.max(...bindings.map(([key]) => key.length))
    this.controls = bindings.map(([key, description]) => `  ${key.padEnd(keyWidth)}  ${description}`)
  }

  /** RMS of Laplacian(p) - divergence: how far the pressure solve actually got. */
  setResidual(rms: number): void {
    this.residual = rms
  }

  /** First scope is treated as the frame total; empty when timestamps aren't available. */
  frame(scopes: ScopeTiming[]): void {
    const now = performance.now()
    this.samples.push(now - this.lastFrameTime)
    this.lastFrameTime = now
    if (this.samples.length > SAMPLE_WINDOW) this.samples.shift()

    if (now - this.lastUpdate < UPDATE_INTERVAL_MS) return
    this.lastUpdate = now

    const mean = this.samples.reduce((sum, v) => sum + v, 0) / this.samples.length
    const period = refreshPeriodMs(this.samples)
    const lines = [`${(1000 / mean).toFixed(0)} fps  ${mean.toFixed(1)} ms frame`]

    if (scopes.length === 0) {
      lines.push('GPU timing unavailable (no timestamp-query)')
    } else {
      const [total, ...stages] = scopes
      const budgetShare = (total.ms / period) * 100
      // Whatever the limiter is, it isn't the GPU while the GPU sits idle most of the period.
      const verdict =
        budgetShare >= 80
          ? 'GPU-limited'
          : mean > period * 1.25
            ? 'stalling outside the GPU'
            : `vsync-limited, GPU using ${budgetShare.toFixed(0)}% of the budget`
      lines.push(`${total.ms.toFixed(2)} ms GPU  ${verdict}`, '')
      const labelWidth = Math.max(...stages.map((s) => s.name.length))
      for (const stage of stages) {
        const stageShare = total.ms > 0 ? (stage.ms / total.ms) * 100 : 0
        lines.push(
          `  ${stage.name.padEnd(labelWidth)}  ${stage.ms.toFixed(2).padStart(5)} ms  ${stageShare.toFixed(0).padStart(3)}%`,
        )
      }
    }

    if (this.residual !== null) lines.push('', `residual ${this.residual.toExponential(2)}`)
    if (this.extraLine) lines.push(this.extraLine)
    if (this.controls.length > 0) lines.push('', 'controls', ...this.controls)
    this.element.textContent = lines.join('\n')
  }

  toggle(): void {
    this.element.hidden = !this.element.hidden
  }
}
