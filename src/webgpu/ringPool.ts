/**
 * Reusable scratch resources, one slot per call. A pass can't share a single scratch texture
 * across calls in the same encoder, but allocating per call churns memory.
 *
 * `size` must exceed the calls a pass makes per frame.
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
 * Same idea, but for uniforms, and skips the upload when a slot already holds the same values.
 * Most params are constant every frame, so after the ring fills they stop uploading entirely.
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
