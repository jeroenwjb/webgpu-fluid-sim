const READBACK_POOL_SIZE = 3
const SMOOTHING = 0.1 // raw per-frame timings are noisy

/**
 * Per-stage GPU timing. Timestamps can only be written via `timestampWrites` on a pass
 * descriptor, so rather than threading that through every pass (diffusion and projection run
 * 20-60 internally), each scope is bracketed by two empty compute passes carrying them.
 *
 * Those marker passes cost ~0.066 ms to schedule, which sets a floor: stages cheaper than
 * that all read 0.066 and every scope is inflated by roughly the same. Fine for finding which
 * stage dominates, not for costing individual cheap passes.
 */
export class GpuProfiler {
  private scopeNames: string[]
  private querySet: GPUQuerySet
  private resolveBuffer: GPUBuffer
  private pool: { buffer: GPUBuffer; inUse: boolean }[] = []
  private pending: { buffer: GPUBuffer; inUse: boolean } | null = null
  private results = new Map<string, number>()
  private queryCount: number
  private device: GPUDevice
  private recordBuffer: GPUBuffer | null = null
  private recordReadback: GPUBuffer | null = null
  private recordStride = 0
  private recordFrames = 0

  constructor(device: GPUDevice, scopeNames: string[]) {
    this.device = device
    this.scopeNames = scopeNames
    this.queryCount = scopeNames.length * 2

    this.querySet = device.createQuerySet({ type: 'timestamp', count: this.queryCount })
    this.resolveBuffer = device.createBuffer({
      size: this.queryCount * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    for (let i = 0; i < READBACK_POOL_SIZE; i++) {
      this.pool.push({
        buffer: device.createBuffer({
          size: this.queryCount * 8,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        inUse: false,
      })
    }
  }

  /** Smoothed ms per scope, in declaration order. */
  get timings(): { name: string; ms: number }[] {
    return this.scopeNames.map((name) => ({ name, ms: this.results.get(name) ?? 0 }))
  }

  begin(encoder: GPUCommandEncoder, scope: string): void {
    const index = this.scopeNames.indexOf(scope)
    if (index < 0) return
    encoder
      .beginComputePass({
        timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: index * 2 },
      })
      .end()
  }

  end(encoder: GPUCommandEncoder, scope: string): void {
    const index = this.scopeNames.indexOf(scope)
    if (index < 0) return
    encoder
      .beginComputePass({
        timestampWrites: { querySet: this.querySet, endOfPassWriteIndex: index * 2 + 1 },
      })
      .end()
  }

  /**
   * Recording mode for benchmarks: each frame's timestamps go into their own slot of one
   * buffer and nothing is mapped until the run ends, so no readback perturbs the measurement.
   * resolveQuerySet needs a 256-byte-aligned offset, hence the padded stride.
   */
  startRecording(frames: number): void {
    this.stopRecording()
    this.recordStride = Math.ceil((this.queryCount * 8) / 256) * 256
    this.recordBuffer = this.device.createBuffer({
      size: this.recordStride * frames,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    this.recordReadback = this.device.createBuffer({
      size: this.recordStride * frames,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
    this.recordFrames = frames
  }

  /** Records one frame's timings. Call instead of resolve() while benchmarking. */
  record(encoder: GPUCommandEncoder, frameIndex: number): void {
    if (!this.recordBuffer || frameIndex >= this.recordFrames) return
    encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.recordBuffer, frameIndex * this.recordStride)
  }

  /** Copies the recording out and returns per-frame, per-scope milliseconds. */
  async finishRecording(): Promise<{ name: string; samples: number[] }[] | null> {
    if (!this.recordBuffer || !this.recordReadback) return null

    const encoder = this.device.createCommandEncoder()
    encoder.copyBufferToBuffer(this.recordBuffer, 0, this.recordReadback, 0, this.recordReadback.size)
    this.device.queue.submit([encoder.finish()])

    await this.recordReadback.mapAsync(GPUMapMode.READ)
    const raw = new BigUint64Array(this.recordReadback.getMappedRange().slice(0))
    this.recordReadback.unmap()

    const stridePairs = this.recordStride / 8
    const results = this.scopeNames.map((name) => ({ name, samples: [] as number[] }))
    for (let frame = 0; frame < this.recordFrames; frame++) {
      const base = frame * stridePairs
      this.scopeNames.forEach((_, scope) => {
        const start = raw[base + scope * 2]
        const end = raw[base + scope * 2 + 1]
        if (end > start) results[scope].samples.push(Number(end - start) / 1e6)
      })
    }

    this.stopRecording()
    return results
  }

  private stopRecording(): void {
    this.recordBuffer?.destroy()
    this.recordReadback?.destroy()
    this.recordBuffer = null
    this.recordReadback = null
  }

  /** Call once per frame, before submitting. */
  resolve(encoder: GPUCommandEncoder): void {
    const slot = this.pool.find((entry) => !entry.inUse && entry.buffer.mapState === 'unmapped')
    if (!slot) {
      this.pending = null
      return
    }

    slot.inUse = true
    this.pending = slot
    encoder.resolveQuerySet(this.querySet, 0, this.queryCount, this.resolveBuffer, 0)
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, slot.buffer, 0, this.queryCount * 8)
  }

  /** Must run after queue.submit() - mapping a buffer an unsubmitted encoder still references throws. */
  readback(): void {
    const slot = this.pending
    this.pending = null
    if (!slot) return

    slot.buffer
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const times = new BigUint64Array(slot.buffer.getMappedRange().slice(0))
        slot.buffer.unmap()

        this.scopeNames.forEach((name, i) => {
          const start = times[i * 2]
          const end = times[i * 2 + 1]
          if (end <= start) return // scope wasn't encoded this frame
          const ms = Number(end - start) / 1e6 // nanoseconds
          const previous = this.results.get(name)
          this.results.set(name, previous === undefined ? ms : previous + (ms - previous) * SMOOTHING)
        })
      })
      .catch(() => {
        // Device lost or buffer destroyed - drop it.
      })
      .finally(() => {
        slot.inUse = false
      })
  }
}
