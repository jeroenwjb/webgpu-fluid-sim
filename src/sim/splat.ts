import { createComputePipeline } from '../webgpu/computePipeline'
import splatWGSL from '../shaders/splat.wgsl?raw'
import { UniformRing } from '../webgpu/ringPool'
import type { Field } from './field'

// Velocity and dye are splatted once each per frame; slack for future callers.
const POOL_SIZE = 4

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
  private uniforms: UniformRing

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, splatWGSL)
    // Pooled, not shared: queue.writeBuffer() takes effect at submit time, so one shared
    // buffer would let a later call's params clobber an earlier one in the same encoder.
    // Unlike the other passes these genuinely change per frame, but only while dragging.
    this.uniforms = new UniformRing(device, POOL_SIZE, 32)
  }

  destroy(): void {
    this.uniforms.destroy()
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: SplatParams): void {
    const uniformBuffer = this.uniforms.write(
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
        { binding: 0, resource: { buffer: uniformBuffer } },
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
