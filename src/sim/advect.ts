import { createComputePipeline } from '../webgpu/computePipeline'
import advectWGSL from '../shaders/advect.wgsl?raw'
import type { Field } from './field'

export interface AdvectParams {
  dt: number
  /** Per-frame decay of the advected value; 1 = none. */
  dissipation: number
}

export class AdvectPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, advectWGSL)
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
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
    // Per-call buffer: velocity and dye advection use different dissipation in the same
    // encoder, and queue.writeBuffer() only lands at submit time - a shared buffer would give
    // both passes whichever value was written last.
    const uniformBuffer = this.device.createBuffer({
      size: 8, // f32 dt + f32 dissipation
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([params.dt, params.dissipation]))

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
