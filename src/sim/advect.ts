import { createComputePipeline } from '../webgpu/computePipeline'
import advectWGSL from '../shaders/advect.wgsl?raw'
import { UniformRing } from '../webgpu/ringPool'
import type { Field } from './field'

// Velocity and dye are each advected once per frame; slack for future callers.
const POOL_SIZE = 4

export interface AdvectParams {
  dt: number
  /** Per-frame decay of the advected value; 1 = none. */
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
    // Pooled, not shared: velocity and dye advection use different dissipation in the same
    // encoder, and queue.writeBuffer() only lands at submit time. These params are constant
    // frame to frame, so the ring's dirty check makes the uploads stop after a few frames.
    this.uniforms = new UniformRing(device, POOL_SIZE, 8) // f32 dt + f32 dissipation
  }

  destroy(): void {
    this.uniforms.destroy()
  }

  // velocityField determines the backward trace; sourceField is the field being carried
  // (pass the same Field for both to self-advect a velocity field).
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
