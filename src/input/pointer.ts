export interface PointerState {
  isDown: boolean
  /** Position in sim texel space. */
  x: number
  y: number
  /** Movement since the last frame, in sim texels. */
  dx: number
  dy: number
  /** True if the pointer moved since the last frame while held down. */
  moved: boolean
}

/**
 * Tracks pointer position/delta in simulation texel space. Deliberately WebGPU-unaware:
 * it only mutates plain state, which the simulation loop samples once per frame.
 *
 * Pointer Events unify mouse/touch/pen, so touch needs no separate path (the canvas also
 * sets `touch-action: none` so dragging doesn't scroll the page).
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

  /** Reads the current state, then clears the per-frame delta. */
  consume(): PointerState {
    const snapshot = { ...this.state }
    this.state.dx = 0
    this.state.dy = 0
    this.state.moved = false
    return snapshot
  }

  private toSimCoords(e: PointerEvent): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect()
    // Texture row 0 renders at the top of the screen (fullscreenQuad.wgsl flips V), and
    // clientY also grows downward, so no Y flip is needed here.
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
    // Seed last position on press so the first move can't produce a huge bogus delta.
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

    // Accumulate, since several pointermove events can arrive between frames.
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
