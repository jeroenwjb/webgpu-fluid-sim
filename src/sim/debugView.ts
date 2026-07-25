import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import { Reducer } from '../webgpu/reducer'
import colorizeSignedWGSL from '../shaders/colorizeSigned.wgsl?raw'
import colorizeVelocityWGSL from '../shaders/colorizeVelocity.wgsl?raw'

export type DebugField = 'dye' | 'velocity' | 'divergence' | 'pressure' | 'curl'

export const DEBUG_FIELDS: DebugField[] = ['dye', 'velocity', 'divergence', 'pressure', 'curl']

/** Colourises a live sim field for display. Dye is shown directly and skips this. */
export class DebugView {
  private device: GPUDevice
  private signedPipeline: GPUComputePipeline
  private velocityPipeline: GPUComputePipeline
  private target: GPUTexture
  private maxReducer: Reducer
  private lastScale: GPUTexture | null = null

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device
    this.signedPipeline = createComputePipeline(device, colorizeSignedWGSL)
    this.velocityPipeline = createComputePipeline(device, colorizeVelocityWGSL)
    this.target = createStorageTexture(device, width, height)
    // Mean rather than max: divergence spikes hard at the injection point, and a max-based
    // scale lets those few texels squash everything else to black.
    this.maxReducer = new Reducer(device, width, height, 'meanAbs')
  }

  /** 1x1 texture holding the scale the last colorize used, for the legend labels. */
  get scaleTexture(): GPUTexture | null {
    return this.lastScale
  }

  destroy(): void {
    this.target.destroy()
    this.maxReducer.destroy()
  }

  /** Returns the view to display, or null if the source isn't available yet. */
  colorize(
    encoder: GPUCommandEncoder,
    field: DebugField,
    sources: { velocity: GPUTexture; divergence: GPUTexture | null; pressure: GPUTexture; curl: GPUTexture },
    width: number,
    height: number,
  ): GPUTextureView | null {
    const source = field === 'velocity' ? sources.velocity
      : field === 'divergence' ? sources.divergence
      : field === 'pressure' ? sources.pressure
      : sources.curl
    if (!source) return null

    const isVelocity = field === 'velocity'
    const pipeline = isVelocity ? this.velocityPipeline : this.signedPipeline

    const entries: { binding: number; resource: GPUBindingResource }[] = [
      { binding: 0, resource: source.createView() },
    ]
    if (isVelocity) {
      this.lastScale = null
      entries.push({ binding: 1, resource: this.target.createView() })
    } else {
      const scale = this.maxReducer.reduce(encoder, source)
      this.lastScale = scale
      entries.push({ binding: 1, resource: scale.createView() })
      entries.push({ binding: 2, resource: this.target.createView() })
    }

    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })

    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
    pass.end()

    return this.target.createView()
  }
}
