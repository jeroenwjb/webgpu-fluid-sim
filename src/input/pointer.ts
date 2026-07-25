/** All coordinates are in sim texel space. */
export interface PointerState {
  isDown: boolean
  x: number
  y: number
  /** Movement since the last frame. */
  dx: number
  dy: number
  moved: boolean
}

/**
 * Pointer state for the sim loop to read once per frame. Kept WebGPU-unaware.
 *
 * Pointer Events cover mouse/touch/pen, so touch needs no separate path.
 */
export class PointerTracker {
  private state: PointerState = { isDown: false, x: 0, y: 0, dx: 0, dy: 0, moved: false }
  private lastX = 0
  private lastY = 0
  private hasLastPosition = false
  private canvas: HTMLCanvasElement
  private simWidth: number
  private simHeight: number

  constructor(canvas: HTMLCanvasElement, simWidth: number, simHeight: number) {
    this.canvas = canvas
    this.simWidth = simWidth
    this.simHeight = simHeight

    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointermove', this.onPointerMove)
    window.addEventListener('pointerup', this.onPointerUp)
    window.addEventListener('pointercancel', this.onPointerUp)
  }

  setSimSize(simWidth: number, simHeight: number): void {
    this.simWidth = simWidth
    this.simHeight = simHeight
    this.hasLastPosition = false // old coords are in the previous grid's space
  }

  /** Reads state and clears the per-frame delta. */
  consume(): PointerState {
    const snapshot = { ...this.state }
    this.state.dx = 0
    this.state.dy = 0
    this.state.moved = false
    return snapshot
  }

  private toSimCoords(e: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect()
    // No Y flip needed: fullscreenQuad.wgsl already flips V, so texture row 0 is at the top.
    return {
      x: ((e.clientX - rect.left) / rect.width) * this.simWidth,
      y: ((e.clientY - rect.top) / rect.height) * this.simHeight,
    }
  }

  private onPointerDown = (e: PointerEvent) => {
    const { x, y } = this.toSimCoords(e)
    this.state.isDown = true
    this.state.x = x
    this.state.y = y
    // Seed on press, otherwise the first move produces a huge delta.
    this.lastX = x
    this.lastY = y
    this.hasLastPosition = true
    this.canvas.setPointerCapture(e.pointerId)
  }

  private onPointerMove = (e: PointerEvent) => {
    if (!this.state.isDown) return

    const { x, y } = this.toSimCoords(e)
    if (!this.hasLastPosition) {
      this.lastX = x
      this.lastY = y
      this.hasLastPosition = true
      return
    }

    // Accumulate - several pointermove events can arrive per frame.
    this.state.dx += x - this.lastX
    this.state.dy += y - this.lastY
    this.state.x = x
    this.state.y = y
    this.state.moved = true
    this.lastX = x
    this.lastY = y
  }

  private onPointerUp = () => {
    this.state.isDown = false
    this.hasLastPosition = false
  }
}
