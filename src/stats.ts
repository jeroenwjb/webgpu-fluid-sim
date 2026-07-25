const SAMPLE_WINDOW = 60
const UPDATE_INTERVAL_MS = 250
const FRAME_BUDGET_MS = 1000 / 60

export interface ScopeTiming {
  name: string
  ms: number
}

/**
 * Performance readout: wall-clock frame time plus per-stage GPU timings.
 *
 * Wall-clock time is vsync-capped, so it shows whether frames are being missed but not how
 * much headroom is left. The GPU timings (from timestamp queries) are the numbers that
 * actually reflect cost.
 */
export class Stats {
  private element: HTMLElement
  private samples: number[] = []
  private lastFrameTime = performance.now()
  private lastUpdate = 0
  private extraLine = ''

  constructor(element: HTMLElement) {
    this.element = element
  }

  /** Extra text shown under the timings, e.g. the current sim resolution. */
  setDetail(text: string): void {
    this.extraLine = text
  }

  /** `scopes` is empty when timestamp queries aren't available. The first scope is the total. */
  frame(scopes: ScopeTiming[]): void {
    const now = performance.now()
    this.samples.push(now - this.lastFrameTime)
    this.lastFrameTime = now
    if (this.samples.length > SAMPLE_WINDOW) this.samples.shift()

    if (now - this.lastUpdate < UPDATE_INTERVAL_MS) return
    this.lastUpdate = now

    const mean = this.samples.reduce((sum, v) => sum + v, 0) / this.samples.length
    const lines = [`${(1000 / mean).toFixed(0)} fps  ${mean.toFixed(1)} ms frame (vsync capped)`]

    if (scopes.length === 0) {
      lines.push('GPU timing unavailable (no timestamp-query)')
    } else {
      const [total, ...stages] = scopes
      lines.push(
        `${total.ms.toFixed(2)} ms GPU  ${((total.ms / FRAME_BUDGET_MS) * 100).toFixed(0)}% of 60fps budget`,
        '',
      )
      const labelWidth = Math.max(...stages.map((s) => s.name.length))
      for (const stage of stages) {
        const share = total.ms > 0 ? (stage.ms / total.ms) * 100 : 0
        lines.push(
          `  ${stage.name.padEnd(labelWidth)}  ${stage.ms.toFixed(2).padStart(5)} ms  ${share.toFixed(0).padStart(3)}%`,
        )
      }
    }

    if (this.extraLine) lines.push('', this.extraLine)
    this.element.textContent = lines.join('\n')
  }

  toggle(): void {
    this.element.hidden = !this.element.hidden
  }
}
