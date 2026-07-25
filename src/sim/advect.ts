import { createComputePipeline } from '../webgpu/computePipeline'
import advectWGSL from '../shaders/advect.wgsl?raw'
import { UniformRing } from '../webgpu/ringPool'
import type { Field } from './field'

const POOL_SIZE = 4

export interface AdvectParams {
  dt: number
  /** Per-frame decay; 1 = none. */
  dissipation: number
}

export class AdvectPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private sampler: GPUSampler
  private uniforms: UniformRing

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, advectWGSL)
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    this.uniforms = new UniformRing(device, POOL_SIZE, 8) // dt + dissipation
  }

  destroy(): void {
    this.uniforms.destroy()
  }

  // Pass the same Field twice to self-advect velocity.
  apply(
    encoder: GPUCommandEncoder,
    velocityField: Field,
    sourceField: Field,
    width: number,
    height: number,
    params: AdvectParams,
  ): void {
    const uniformBuffer = this.uniforms.write(new Float32Array([params.dt, params.dissipation]))

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: velocityField.read.createView() },
        { binding: 3, resource: sourceField.read.createView() },
        { binding: 4, resource: sourceField.write.createView() },
      ],
    })

    const pass = encoder.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
    pass.end()

    sourceField.swap()
  }
}
