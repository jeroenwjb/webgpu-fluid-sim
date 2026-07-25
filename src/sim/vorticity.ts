import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import curlWGSL from '../shaders/curl.wgsl?raw'
import vorticityWGSL from '../shaders/vorticity.wgsl?raw'
import { UniformRing } from '../webgpu/ringPool'
import type { Field } from './field'

const POOL_SIZE = 2

export interface VorticityParams {
  strength: number
  dt: number
}

/** Re-injects the small-scale swirl that semi-Lagrangian advection smears away. */
export class VorticityPass {
  private device: GPUDevice
  private curlPipeline: GPUComputePipeline
  private confinePipeline: GPUComputePipeline
  private curlTexture: GPUTexture
  private uniforms: UniformRing

  /** For the debug view. */
  get curl(): GPUTexture {
    return this.curlTexture
  }

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device
    this.curlPipeline = createComputePipeline(device, curlWGSL)
    this.confinePipeline = createComputePipeline(device, vorticityWGSL)
    this.curlTexture = createStorageTexture(device, width, height)
    this.uniforms = new UniformRing(device, POOL_SIZE, 8) // strength + dt
  }

  /** Call before reallocating at a new resolution. */
  destroy(): void {
    this.curlTexture.destroy()
    this.uniforms.destroy()
  }

  apply(
    encoder: GPUCommandEncoder,
    velocityField: Field,
    width: number,
    height: number,
    params: VorticityParams,
  ): void {
    const workgroupsX = Math.ceil(width / 8)
    const workgroupsY = Math.ceil(height / 8)

    const dispatch = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; resource: GPUBindingResource }[],
    ) => {
      const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupsX, workgroupsY)
      pass.end()
    }

    dispatch(this.curlPipeline, [
      { binding: 0, resource: velocityField.read.createView() },
      { binding: 1, resource: this.curlTexture.createView() },
    ])

    const uniformBuffer = this.uniforms.write(new Float32Array([params.strength, params.dt]))
    dispatch(this.confinePipeline, [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: velocityField.read.createView() },
      { binding: 2, resource: this.curlTexture.createView() },
      { binding: 3, resource: velocityField.write.createView() },
    ])

    velocityField.swap()
  }
}
