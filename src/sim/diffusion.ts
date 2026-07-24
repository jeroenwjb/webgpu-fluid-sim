import { createComputePipeline } from '../webgpu/computePipeline'
import jacobiWGSL from '../shaders/jacobi.wgsl?raw'
import type { Field } from './field'

export interface DiffuseParams {
  alpha: number
  rBeta: number
  iterations: number
}

export class DiffusionPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private uniformBuffer: GPUBuffer
  private sourceTexture: GPUTexture

  constructor(device: GPUDevice, width: number, height: number, format: GPUTextureFormat = 'rgba16float') {
    this.device = device
    this.pipeline = createComputePipeline(device, jacobiWGSL)
    this.uniformBuffer = device.createBuffer({
      size: 8, // f32 alpha + f32 rBeta
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.sourceTexture = device.createTexture({
      size: { width, height },
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: DiffuseParams): void {
    this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([params.alpha, params.rBeta]))

    // Snapshot the field's current value as the fixed source term (b) for every iteration.
    encoder.copyTextureToTexture({ texture: field.read }, { texture: this.sourceTexture }, { width, height })

    for (let i = 0; i < params.iterations; i++) {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: field.read.createView() },
          { binding: 2, resource: this.sourceTexture.createView() },
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
}
