import { createComputePipeline } from '../webgpu/computePipeline'
import advectWGSL from '../shaders/advect.wgsl?raw'
import type { Field } from './field'

export interface AdvectParams {
  dt: number
  angularSpeed: number
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
      size: 8, // f32 dt + f32 angularSpeed
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: AdvectParams): void {
    this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([params.dt, params.angularSpeed]))

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: field.read.createView() },
        { binding: 3, resource: field.write.createView() },
      ],
    })

    const pass = encoder.beginComputePass()
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
    pass.end()

    field.swap()
  }
}
