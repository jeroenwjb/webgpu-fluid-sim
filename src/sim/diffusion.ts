import { createComputePipeline } from '../webgpu/computePipeline'
import jacobiWGSL from '../shaders/jacobi.wgsl?raw'
import { RingPool, UniformRing } from '../webgpu/ringPool'
import type { Field } from './field'

// Dye and velocity each diffuse once per frame; a little slack for future callers.
const POOL_SIZE = 4

export interface DiffuseParams {
  alpha: number
  rBeta: number
  iterations: number
}

export class DiffusionPass {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private uniforms: UniformRing
  private sourceTextures: RingPool<GPUTexture>

  constructor(device: GPUDevice, width: number, height: number, format: GPUTextureFormat = 'rgba16float') {
    this.device = device
    this.pipeline = createComputePipeline(device, jacobiWGSL)

    // Pooled rather than allocated per call: the snapshot texture is full-size, so allocating
    // one per call churned megabytes per frame.
    this.uniforms = new UniformRing(device, POOL_SIZE, 8) // f32 alpha + f32 rBeta
    this.sourceTextures = new RingPool(POOL_SIZE, () =>
      device.createTexture({
        size: { width, height },
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      }),
    )
  }

  /** Frees pooled resources; call when the sim is reallocated at a new resolution. */
  destroy(): void {
    this.uniforms.destroy()
    this.sourceTextures.destroy((texture) => texture.destroy())
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number, params: DiffuseParams): void {
    const uniformBuffer = this.uniforms.write(new Float32Array([params.alpha, params.rBeta]))

    const sourceTexture = this.sourceTextures.next()

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
