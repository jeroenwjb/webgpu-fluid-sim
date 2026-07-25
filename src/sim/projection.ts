import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import divergenceWGSL from '../shaders/divergence.wgsl?raw'
import jacobiWGSL from '../shaders/jacobi.wgsl?raw'
import gradientSubtractWGSL from '../shaders/gradientSubtract.wgsl?raw'
import subtractMeanWGSL from '../shaders/subtractMean.wgsl?raw'
import { RingPool } from '../webgpu/ringPool'
import { Reducer } from '../webgpu/reducer'
import { Field } from './field'

const POOL_SIZE = 2

export interface ProjectParams {
  iterations: number
}

export class ProjectionPass {
  private device: GPUDevice
  private divergenceRawPipeline: GPUComputePipeline
  private jacobiPipeline: GPUComputePipeline
  private gradientSubtractPipeline: GPUComputePipeline
  private pressureField: Field
  private uniformBuffer: GPUBuffer
  private divergenceTextures: RingPool<GPUTexture>
  private lastDivergence: GPUTexture | null = null
  private subtractMeanPipeline: GPUComputePipeline
  private reducer: Reducer
  private divergenceScratch: RingPool<GPUTexture>

  /** Live fields from the most recent solve, for the debug views. */
  get divergenceTexture(): GPUTexture | null {
    return this.lastDivergence
  }
  get pressureTexture(): GPUTexture {
    return this.pressureField.read
  }

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device
    this.divergenceRawPipeline = createComputePipeline(device, divergenceWGSL, 'raw')
    this.jacobiPipeline = createComputePipeline(device, jacobiWGSL)
    this.gradientSubtractPipeline = createComputePipeline(device, gradientSubtractWGSL)
    this.subtractMeanPipeline = createComputePipeline(device, subtractMeanWGSL)
    this.reducer = new Reducer(device, width, height)

    // Kept across frames - warm-starting converges much faster than restarting from zero.
    this.pressureField = new Field(device, width, height)
    this.divergenceTextures = new RingPool(POOL_SIZE, () => createStorageTexture(device, width, height))
    this.divergenceScratch = new RingPool(POOL_SIZE, () => createStorageTexture(device, width, height))

    // Laplacian(p) = div  ->  p = (neighbours - div) / 4
    this.uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([-1, 0.25]))
  }

  /** Call before reallocating at a new resolution. */
  destroy(): void {
    this.divergenceTextures.destroy((texture) => texture.destroy())
    this.divergenceScratch.destroy((texture) => texture.destroy())
    this.reducer.destroy()
  }

  apply(encoder: GPUCommandEncoder, velocityField: Field, width: number, height: number, params: ProjectParams): void {
    const workgroupsX = Math.ceil(width / 8)
    const workgroupsY = Math.ceil(height / 8)

    const divergenceTex = this.divergenceTextures.next()
    this.lastDivergence = divergenceTex

    const dispatch = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; resource: GPUBindingResource }[],
    ) => {
      const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupsX, workgroupsY)
      pass.end()
    }

    const rawDivergence = this.divergenceScratch.next()
    dispatch(this.divergenceRawPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: rawDivergence.createView() },
    ])

    // With free-slip everywhere the Poisson problem is only solvable if the divergence sums
    // to zero. The one-sided stencils clamp on opposite edges for divergence and gradient, so
    // the discrete sum lands slightly off even when the flow is physically closed - and the
    // leftover forces a domain-scale ramp into the pressure. Subtracting the mean makes the
    // source exactly compatible.
    const divergenceMean = this.reducer.reduce(encoder, rawDivergence)
    dispatch(this.subtractMeanPipeline, [
      { binding: 0, resource: rawDivergence.createView() },
      { binding: 1, resource: divergenceMean.createView() },
      { binding: 2, resource: divergenceTex.createView() },
    ])

    for (let i = 0; i < params.iterations; i++) {
      dispatch(this.jacobiPipeline, [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.pressureField.read.createView() },
        { binding: 2, resource: divergenceTex.createView() },
        { binding: 3, resource: this.pressureField.write.createView() },
      ])
      this.pressureField.swap()
    }

    // Neumann boundaries leave pressure defined only up to a constant, and warm-starting
    // means nothing pins it down - it random-walks and would eventually saturate fp16.
    // Invisible to the flow either way, since only the gradient is used.
    const mean = this.reducer.reduce(encoder, this.pressureField.read)
    dispatch(this.subtractMeanPipeline, [
      { binding: 0, resource: this.pressureField.read.createView() },
      { binding: 1, resource: mean.createView() },
      { binding: 2, resource: this.pressureField.write.createView() },
    ])
    this.pressureField.swap()

    dispatch(this.gradientSubtractPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: this.pressureField.read.createView() },
      { binding: 2, resource: velocityField.write.createView() },
    ])

    velocityField.swap()
  }
}
