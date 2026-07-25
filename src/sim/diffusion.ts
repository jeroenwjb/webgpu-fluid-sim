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
  private format: GPUTextureFormat

  constructor(device: GPUDevice, _width: number, _height: number, format: GPUTextureFormat = 'rgba16float') {
    this.device = device
    this.pipeline = createComputePipeline(device, jacobiWGSL)
    this.format = format
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: DiffuseParams): void {
    // Fresh per-call resources: queue.writeBuffer() lands at submit time and the snapshot
    // texture is written by the encoder, so sharing either across calls recorded into the
    // same encoder would let a later call clobber an earlier one.
    const uniformBuffer = this.device.createBuffer({
      size: 8, // f32 alpha + f32 rBeta
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([params.alpha, params.rBeta]))

    const sourceTexture = this.device.createTexture({
      size: { width, height },
      format: this.format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })

    // Snapshot the field's current value as the fixed source term (b) for every iteration.
    encoder.copyTextureToTexture({ texture: field.read }, { texture: sourceTexture }, { width, height })

    for (let i = 0; i < params.iterations; i++) {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: field.read.createView() },
          { binding: 2, resource: sourceTexture.createView() },
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
