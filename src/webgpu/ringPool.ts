/**
 * Fixed-size ring of reusable GPU resources.
 *
 * Passes can't share a single scratch buffer/texture across calls recorded into the same
 * encoder - `queue.writeBuffer()` only lands at submit time, and encoder writes land in
 * submission order, so a later call would clobber an earlier one. Allocating fresh resources
 * per call avoids that but churns memory (full-size textures every frame).
 *
 * A ring gives each call within a frame a distinct slot while reusing across frames. Reuse is
 * safe because queue operations are ordered: next frame's write is sequenced after the prior
 * frame's submitted work that read it.
 *
 * `size` must exceed the maximum number of calls a single pass makes per frame.
 */
export class RingPool<T> {
  private items: T[]
  private index = 0

  constructor(size: number, create: () => T) {
    this.items = Array.from({ length: size }, create)
  }

  next(): T {
    const item = this.items[this.index]
    this.index = (this.index + 1) % this.items.length
    return item
  }

  destroy(destroyItem: (item: T) => void): void {
    this.items.forEach(destroyItem)
    this.items = []
  }
}

/**
 * Ring of uniform buffers that skips redundant uploads.
 *
 * Most pass parameters (diffusion alpha/rBeta, advection dt/dissipation) are identical every
 * frame, so re-uploading them each frame is pure waste. Each slot remembers its last contents
 * and only writes when they actually change: constant parameters settle after one pass through
 * the ring and then transfer nothing, while genuinely dynamic ones (splat position/colour)
 * still upload only when they change.
 */
export class UniformRing {
  private device: GPUDevice
  private buffers: GPUBuffer[]
  private cached: (Float32Array | null)[]
  private index = 0

  constructor(device: GPUDevice, size: number, byteLength: number) {
    this.device = device
    this.buffers = Array.from({ length: size }, () =>
      device.createBuffer({
        size: byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    )
    this.cached = Array.from({ length: size }, () => null)
  }

  /** Returns a buffer holding `values`, uploading only if this slot doesn't already match. */
  write(values: Float32Array): GPUBuffer {
    const slot = this.index
    this.index = (this.index + 1) % this.buffers.length

    const previous = this.cached[slot]
    if (!previous || !equal(previous, values)) {
      this.device.queue.writeBuffer(this.buffers[slot], 0, values)
      this.cached[slot] = values.slice()
    }
    return this.buffers[slot]
  }

  destroy(): void {
    this.buffers.forEach((buffer) => buffer.destroy())
    this.buffers = []
    this.cached = []
  }
}

function equal(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false
  }
  return true
}
