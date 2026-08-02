import {
  createComputePipeline,
  createStorageTexture,
  SCALAR_FORMAT,
  asScalarShader,
} from '../../webgpu/computePipeline'
import { dispatch } from '../../webgpu/dispatch'
import redBlackWGSL from '../../shaders/redBlackGaussSeidel.wgsl?raw'
import residualWGSL from '../../shaders/residual.wgsl?raw'
import restrictWGSL from '../../shaders/restrict.wgsl?raw'
import prolongWGSL from '../../shaders/prolong.wgsl?raw'
import clearWGSL from '../../shaders/clear.wgsl?raw'
import type { PressureSolver } from './pressureSolver'
import type { Field } from '../field'

const COARSEST = 8 // stop here; below this the solve is trivial and the dispatch overhead dominates
const PRE_SMOOTH = 2
const POST_SMOOTH = 2
const COARSEST_SMOOTH = 8

interface Level {
  width: number
  height: number
  /** Approximation on this level. On level 0 this is the real pressure, below it the error. */
  x: GPUTexture
  /** Right-hand side: divergence on level 0, restricted residual below. */
  b: GPUTexture
  residual: GPUTexture
}

/**
 * Multigrid V-cycle.
 *
 * Smoothers kill high-frequency error quickly and crawl on smooth error - measured directly
 * here, where 60 -> 600 Jacobi iterations barely moved the residual. Multigrid sidesteps that:
 * smooth error looks high-frequency on a coarser grid, so it gets solved cheaply there and
 * carried back as a correction.
 *
 * The coarse grids solve for the ERROR, not the pressure. If p~ is the current approximation,
 * the error e = p - p~ satisfies A.e = b - A.p~ = residual - same operator, and e is smooth
 * after smoothing, which is exactly what a coarse grid handles well.
 */
export class MultigridSolver implements PressureSolver {
  readonly name = 'multigrid'
  private device: GPUDevice
  private smoothPipeline: GPUComputePipeline
  private residualPipeline: GPUComputePipeline
  private restrictPipeline: GPUComputePipeline
  private prolongPipeline: GPUComputePipeline
  private clearPipeline: GPUComputePipeline
  private redBuffer: GPUBuffer
  private blackBuffer: GPUBuffer
  private levels: Level[] = []
  private cycles: number

  constructor(device: GPUDevice, width: number, height: number, cycles = 1) {
    this.device = device
    this.cycles = cycles

    this.smoothPipeline = createComputePipeline(device, asScalarShader(redBlackWGSL), 'main', [
      'uniform',
      'texture',
      { storage: SCALAR_FORMAT, access: 'read-write' },
    ])
    this.residualPipeline = createComputePipeline(device, asScalarShader(residualWGSL), 'main', [
      'texture',
      'texture',
      { storage: SCALAR_FORMAT },
    ])
    this.restrictPipeline = createComputePipeline(device, asScalarShader(restrictWGSL), 'main', [
      'texture',
      { storage: SCALAR_FORMAT },
    ])
    this.prolongPipeline = createComputePipeline(device, asScalarShader(prolongWGSL), 'main', [
      'texture',
      { storage: SCALAR_FORMAT, access: 'read-write' },
    ])
    this.clearPipeline = createComputePipeline(device, asScalarShader(clearWGSL), 'main', [
      { storage: SCALAR_FORMAT },
    ])

    const makeParams = (parity: number) => {
      const buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(buffer, 0, new Float32Array([-1, 0.25, parity, 0]))
      return buffer
    }
    this.redBuffer = makeParams(0)
    this.blackBuffer = makeParams(1)

    // Level 0 borrows the caller's pressure and divergence, so only its residual is allocated.
    let w = width
    let h = height
    let first = true
    while (true) {
      this.levels.push({
        width: w,
        height: h,
        x: first ? (null as unknown as GPUTexture) : createStorageTexture(device, w, h, SCALAR_FORMAT),
        b: first ? (null as unknown as GPUTexture) : createStorageTexture(device, w, h, SCALAR_FORMAT),
        residual: createStorageTexture(device, w, h, SCALAR_FORMAT),
      })
      first = false
      if (w <= COARSEST && h <= COARSEST) break
      w = Math.max(1, Math.ceil(w / 2))
      h = Math.max(1, Math.ceil(h / 2))
    }
  }

  destroy(): void {
    this.redBuffer.destroy()
    this.blackBuffer.destroy()
    for (const level of this.levels) {
      level.x?.destroy()
      level.b?.destroy()
      level.residual.destroy()
    }
    this.levels = []
  }

  solve(
    encoder: GPUCommandEncoder,
    pressure: Field,
    divergence: GPUTexture,
    _width: number,
    _height: number,
  ): void {
    for (let i = 0; i < this.cycles; i++) {
      this.vCycle(encoder, 0, pressure.read, divergence)
    }
  }

  private smooth(encoder: GPUCommandEncoder, x: GPUTexture, b: GPUTexture, level: Level, sweeps: number): void {
    for (let i = 0; i < sweeps; i++) {
      for (const params of [this.redBuffer, this.blackBuffer]) {
        dispatch(
          this.device,
          encoder,
          this.smoothPipeline,
          [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: b.createView() },
            { binding: 2, resource: x.createView() },
          ],
          level.width,
          level.height,
        )
      }
    }
  }

  private vCycle(encoder: GPUCommandEncoder, depth: number, x: GPUTexture, b: GPUTexture): void {
    const level = this.levels[depth]
    const coarsest = depth === this.levels.length - 1

    if (coarsest) {
      this.smooth(encoder, x, b, level, COARSEST_SMOOTH)
      return
    }

    this.smooth(encoder, x, b, level, PRE_SMOOTH)

    dispatch(
      this.device,
      encoder,
      this.residualPipeline,
      [
        { binding: 0, resource: x.createView() },
        { binding: 1, resource: b.createView() },
        { binding: 2, resource: level.residual.createView() },
      ],
      level.width,
      level.height,
    )

    const below = this.levels[depth + 1]
    dispatch(
      this.device,
      encoder,
      this.restrictPipeline,
      [
        { binding: 0, resource: level.residual.createView() },
        { binding: 1, resource: below.b.createView() },
      ],
      below.width,
      below.height,
    )

    // The coarse grid solves for a correction, so it has to start from zero - otherwise it
    // keeps adding last frame's error on top of this one's.
    dispatch(
      this.device,
      encoder,
      this.clearPipeline,
      [{ binding: 0, resource: below.x.createView() }],
      below.width,
      below.height,
    )

    this.vCycle(encoder, depth + 1, below.x, below.b)

    dispatch(
      this.device,
      encoder,
      this.prolongPipeline,
      [
        { binding: 0, resource: below.x.createView() },
        { binding: 1, resource: x.createView() },
      ],
      level.width,
      level.height,
    )

    this.smooth(encoder, x, b, level, POST_SMOOTH)
  }
}
