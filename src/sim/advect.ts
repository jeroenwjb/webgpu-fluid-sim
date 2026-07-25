import { createComputePipeline } from '../webgpu/computePipeline'
import advectWGSL from '../shaders/advect.wgsl?raw'
import type { Field } from './field'

export interface AdvectParams {
  dt: number
}

export class AdvectPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private uniformBuffer: GPUBuffer
  private sampler: GPUSampler

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, advectWGSL)
    this.uniformBuffer = device.createBuffer({
      size: 4, // f32 dt
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
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
    this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([params.dt]))

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
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
