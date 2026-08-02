import { createComputePipeline, createStorageTexture, SCALAR_FORMAT } from '../webgpu/computePipeline'
import divergenceWGSL from '../shaders/divergence.wgsl?raw'
import gradientSubtractWGSL from '../shaders/gradientSubtract.wgsl?raw'
import subtractMeanWGSL from '../shaders/subtractMean.wgsl?raw'
import residualWGSL from '../shaders/residual.wgsl?raw'
import { RingPool } from '../webgpu/ringPool'
import { Reducer } from '../webgpu/reducer'
import { dispatch } from '../webgpu/dispatch'
import { Field } from './field'
import type { PressureSolver } from './solvers/pressureSolver'

const POOL_SIZE = 2

export class ProjectionPass {
  private device: GPUDevice
  private divergenceRawPipeline: GPUComputePipeline
  private gradientSubtractPipeline: GPUComputePipeline
  private pressureField: Field
  private solver: PressureSolver
  private divergenceTextures: RingPool<GPUTexture>
  private lastDivergence: GPUTexture | null = null
  private subtractMeanPipeline: GPUComputePipeline
  private reducer: Reducer
  private divergenceScratch: RingPool<GPUTexture>
  private residualPipeline: GPUComputePipeline
  private residualTexture: GPUTexture

  /** Signed residual field from the last measureResidual() call, for the debug view. */
  get residual(): GPUTexture {
    return this.residualTexture
  }

  /** Live fields from the most recent solve, for the debug views. */
  get divergenceTexture(): GPUTexture | null {
    return this.lastDivergence
  }
  get pressureTexture(): GPUTexture {
    return this.pressureField.read
  }

  constructor(device: GPUDevice, width: number, height: number, solver: PressureSolver) {
    this.device = device
    this.solver = solver
    this.divergenceRawPipeline = createComputePipeline(device, divergenceWGSL, 'raw', [
      'texture',
      { storage: SCALAR_FORMAT },
    ])
    this.gradientSubtractPipeline = createComputePipeline(device, gradientSubtractWGSL, 'main', [
      'texture',
      'texture',
      { storage: 'rgba16float' },
    ])
    this.subtractMeanPipeline = createComputePipeline(device, subtractMeanWGSL, 'main', [
      'texture',
      'texture',
      { storage: SCALAR_FORMAT },
    ])
    this.reducer = new Reducer(device, width, height)
    this.residualPipeline = createComputePipeline(device, residualWGSL, 'main', [
      'texture',
      'texture',
      { storage: 'rgba16float' },
    ])
    this.residualTexture = createStorageTexture(device, width, height)

    // Kept across frames - warm-starting converges much faster than restarting from zero.
    this.pressureField = new Field(device, width, height, SCALAR_FORMAT)
    this.divergenceTextures = new RingPool(POOL_SIZE, () => createStorageTexture(device, width, height, SCALAR_FORMAT))
    this.divergenceScratch = new RingPool(POOL_SIZE, () => createStorageTexture(device, width, height, SCALAR_FORMAT))
  }

  /** Call before reallocating at a new resolution. */
  destroy(): void {
    this.divergenceTextures.destroy((texture) => texture.destroy())
    this.divergenceScratch.destroy((texture) => texture.destroy())
    this.reducer.destroy()
    this.residualTexture.destroy()
    this.solver.destroy()
  }

  /**
   * Records how far the solve actually got. Returns a 1x1 texture whose .g is the mean square
   * residual - sqrt it for RMS. Costs a dispatch plus a reduction, so call it sparingly.
   */
  measureResidual(encoder: GPUCommandEncoder, width: number, height: number): GPUTexture | null {
    if (!this.lastDivergence) return null

    dispatch(
      this.device,
      encoder,
      this.residualPipeline,
      [
        { binding: 0, resource: this.pressureField.read.createView() },
        { binding: 1, resource: this.lastDivergence.createView() },
        { binding: 2, resource: this.residualTexture.createView() },
      ],
      width,
      height,
    )
    return this.reducer.reduce(encoder, this.residualTexture)
  }

  /** Swaps the solver in place, keeping the warm-started pressure field. */
  setSolver(solver: PressureSolver): void {
    this.solver = solver
  }

  get solverName(): string {
    return this.solver.name
  }

  apply(encoder: GPUCommandEncoder, velocityField: Field, width: number, height: number): void {
    const divergenceTex = this.divergenceTextures.next()
    this.lastDivergence = divergenceTex

    const run = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; resource: GPUBindingResource }[],
    ) => dispatch(this.device, encoder, pipeline, entries, width, height)

    const rawDivergence = this.divergenceScratch.next()
    run(this.divergenceRawPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: rawDivergence.createView() },
    ])

    // With free-slip everywhere the Poisson problem is only solvable if the divergence sums
    // to zero. The one-sided stencils clamp on opposite edges for divergence and gradient, so
    // the discrete sum lands slightly off even when the flow is physically closed - and the
    // leftover forces a domain-scale ramp into the pressure. Subtracting the mean makes the
    // source exactly compatible.
    const divergenceMean = this.reducer.reduce(encoder, rawDivergence)
    run(this.subtractMeanPipeline, [
      { binding: 0, resource: rawDivergence.createView() },
      { binding: 1, resource: divergenceMean.createView() },
      { binding: 2, resource: divergenceTex.createView() },
    ])

    this.solver.solve(encoder, this.pressureField, divergenceTex, width, height)

    // Neumann boundaries leave pressure defined only up to a constant, and warm-starting
    // means nothing pins it down - it random-walks and would eventually saturate fp16.
    // Invisible to the flow either way, since only the gradient is used.
    const mean = this.reducer.reduce(encoder, this.pressureField.read)
    run(this.subtractMeanPipeline, [
      { binding: 0, resource: this.pressureField.read.createView() },
      { binding: 1, resource: mean.createView() },
      { binding: 2, resource: this.pressureField.write.createView() },
    ])
    this.pressureField.swap()

    run(this.gradientSubtractPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: this.pressureField.read.createView() },
      { binding: 2, resource: velocityField.write.createView() },
    ])

    velocityField.swap()
  }
}
