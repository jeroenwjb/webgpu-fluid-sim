import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import divergenceWGSL from '../shaders/divergence.wgsl?raw'
import jacobiWGSL from '../shaders/jacobi.wgsl?raw'
import gradientSubtractWGSL from '../shaders/gradientSubtract.wgsl?raw'
import { Field } from './field'

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

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device
    this.divergenceRawPipeline = createComputePipeline(device, divergenceWGSL, 'raw')
    this.jacobiPipeline = createComputePipeline(device, jacobiWGSL)
    this.gradientSubtractPipeline = createComputePipeline(device, gradientSubtractWGSL)

    // Pressure is warm-started from the previous frame's solution (still zero-initialized on the
    // very first frame) rather than reset to zero every call - converges faster since pressure
    // doesn't change drastically frame to frame.
    this.pressureField = new Field(device, width, height)

    // Poisson pressure equation ∇²p = div, discretized with the compact Laplacian:
    // (pL + pR + pU + pD - 4p) = div  ->  p = (neighbors - div) / 4  ->  alpha=-1, rBeta=1/4.
    this.uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([-1, 0.25]))
  }

  apply(encoder: GPUCommandEncoder, velocityField: Field, width: number, height: number, params: ProjectParams): void {
    const workgroupsX = Math.ceil(width / 8)
    const workgroupsY = Math.ceil(height / 8)

    // Fresh per-call so two projections recorded into one encoder can't clobber each other.
    const divergenceTex = createStorageTexture(this.device, width, height)

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

    dispatch(this.divergenceRawPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: divergenceTex.createView() },
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

    dispatch(this.gradientSubtractPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: this.pressureField.read.createView() },
      { binding: 2, resource: velocityField.write.createView() },
    ])

    velocityField.swap()
  }
}
