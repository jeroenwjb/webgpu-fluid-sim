import { createComputePipeline } from '../webgpu/computePipeline'
import splatWGSL from '../shaders/splat.wgsl?raw'
import type { Field } from './field'

export interface SplatParams {
  x: number
  y: number
  radius: number
  strength: number
  color: [number, number, number, number]
}

export class SplatPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private uniformBuffer: GPUBuffer

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, splatWGSL)
    this.uniformBuffer = device.createBuffer({
      size: 32, // vec2f position + f32 radius + f32 strength + vec4f color
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: SplatParams): void {
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Float32Array([
        params.x,
        params.y,
        params.radius,
        params.strength,
        params.color[0],
        params.color[1],
        params.color[2],
        params.color[3],
      ]),
    )

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: field.read.createView() },
        { binding: 2, resource: field.write.createView() },
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
