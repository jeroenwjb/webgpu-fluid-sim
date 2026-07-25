const SAMPLE_WINDOW = 60
const UPDATE_INTERVAL_MS = 250

export interface ScopeTiming {
  name: string
  ms: number
}

/**
 * Frame time plus per-stage GPU timings. Frame time is vsync-capped so it only shows dropped
 * frames; the GPU numbers are what actually reflect cost.
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

  /** Extra line under the timings, e.g. sim resolution. */
  setDetail(text: string): void {
    this.extraLine = text
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
    const lines = [`${(1000 / mean).toFixed(0)} fps  ${mean.toFixed(1)} ms frame (vsync capped)`]

    if (scopes.length === 0) {
      lines.push('GPU timing unavailable (no timestamp-query)')
    } else {
      const [total, ...stages] = scopes
      const uncapped = total.ms > 0 ? 1000 / total.ms : 0
      lines.push(`${total.ms.toFixed(2)} ms GPU  ~${uncapped.toFixed(0)} fps uncapped`, '')
      const labelWidth = Math.max(...stages.map((s) => s.name.length))
      for (const stage of stages) {
        const stageShare = total.ms > 0 ? (stage.ms / total.ms) * 100 : 0
        lines.push(
          `  ${stage.name.padEnd(labelWidth)}  ${stage.ms.toFixed(2).padStart(5)} ms  ${stageShare.toFixed(0).padStart(3)}%`,
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
