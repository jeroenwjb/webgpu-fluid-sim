import { readSingleTexel } from './debugReadback'

const UPDATE_INTERVAL_FRAMES = 30

/** Colour-bar labels for the auto-scaled debug views. */
export class Legend {
  private root: HTMLElement
  private minLabel: HTMLElement
  private maxLabel: HTMLElement
  private frame = 0
  private pending = false

  constructor(root: HTMLElement, minLabel: HTMLElement, maxLabel: HTMLElement) {
    this.root = root
    this.minLabel = minLabel
    this.maxLabel = maxLabel
  }

  hide(): void {
    this.root.hidden = true
  }

  /**
   * `scaleTexture` is the 1x1 max the view was normalised by. Readback is async and only
   * sampled occasionally - the labels don't need to be frame-accurate.
   */
  update(device: GPUDevice, scaleTexture: GPUTexture | null): void {
    this.root.hidden = scaleTexture === null
    if (!scaleTexture) return

    this.frame++
    if (this.frame % UPDATE_INTERVAL_FRAMES !== 0 || this.pending) return

    this.pending = true
    readSingleTexel(device, scaleTexture)
      .then(([scale]) => {
        const text = formatScale(scale)
        this.minLabel.textContent = `-${text}`
        this.maxLabel.textContent = `+${text}`
      })
      .catch(() => {})
      .finally(() => {
        this.pending = false
      })
  }
}

function formatScale(value: number): string {
  if (!isFinite(value) || value === 0) return '0'
  const abs = Math.abs(value)
  if (abs >= 1000 || abs < 0.01) return abs.toExponential(1)
  return abs.toFixed(abs >= 10 ? 0 : 2)
}
