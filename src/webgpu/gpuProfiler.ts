const READBACK_POOL_SIZE = 3 // enough that a buffer is never still in flight when reused
const SMOOTHING = 0.1 // EMA factor; raw per-frame GPU timings are noisy

/**
 * Per-stage GPU timing via timestamp queries.
 *
 * WebGPU only exposes timestamps through `timestampWrites` on a pass descriptor. Rather than
 * threading that through every pass class - awkward for DiffusionPass/ProjectionPass, which
 * each run 20-40 passes internally - each scope is bracketed by two empty compute passes that
 * carry the timestamps. Everything encoded between them is measured, so scopes can wrap
 * arbitrary groups of work without those classes knowing about profiling at all.
 *
 * This measures real GPU execution time, unlike wall-clock frame time which vsync pins to the
 * refresh rate.
 */
export class GpuProfiler {
  private scopeNames: string[]
  private querySet: GPUQuerySet
  private resolveBuffer: GPUBuffer
  private pool: { buffer: GPUBuffer; inUse: boolean }[] = []
  private pending: { buffer: GPUBuffer; inUse: boolean } | null = null
  private results = new Map<string, number>()
  private queryCount: number

  constructor(device: GPUDevice, scopeNames: string[]) {
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

  /** Smoothed per-scope GPU time in milliseconds, in the order scopes were declared. */
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

  /** Records the resolve + copy. Call once per frame, before submitting. */
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

  /**
   * Maps and reads the resolved timings. Must run AFTER queue.submit() - mapping a buffer
   * that is still referenced by an unsubmitted command buffer is a validation error.
   */
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
          if (end <= start) return // scope not encoded this frame
          const ms = Number(end - start) / 1e6 // timestamps are nanoseconds
          const previous = this.results.get(name)
          this.results.set(name, previous === undefined ? ms : previous + (ms - previous) * SMOOTHING)
        })
      })
      .catch(() => {
        // Device lost or buffer destroyed - drop the sample.
      })
      .finally(() => {
        slot.inUse = false
      })
  }
}
