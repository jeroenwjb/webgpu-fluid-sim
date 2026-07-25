import { createComputePipeline, createStorageTexture } from './computePipeline'
import reduceAverageWGSL from '../shaders/reduceAverage.wgsl?raw'
import reduceMaxAbsWGSL from '../shaders/reduceMaxAbs.wgsl?raw'
import reduceMeanAbsWGSL from '../shaders/reduceMeanAbs.wgsl?raw'

export type ReduceMode = 'average' | 'maxAbs' | 'meanAbs'

const SHADERS: Record<ReduceMode, string> = {
  average: reduceAverageWGSL,
  maxAbs: reduceMaxAbsWGSL,
  meanAbs: reduceMeanAbsWGSL,
}

/**
 * Reduces a texture to 1x1 by repeated 2x2 halving, entirely on the GPU. The result stays in
 * a texture so another pass can consume it without a readback.
 *
 * Odd sizes clamp at the edge, so the last row/column is slightly over-weighted. Fine for
 * removing a drifting constant or auto-scaling a view; not an exact mean.
 */
export class Reducer {
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private levels: GPUTexture[] = []

  constructor(device: GPUDevice, width: number, height: number, mode: ReduceMode = 'average') {
    this.device = device
    this.pipeline = createComputePipeline(device, SHADERS[mode])

    let w = width
    let h = height
    while (w > 1 || h > 1) {
      w = Math.max(1, Math.ceil(w / 2))
      h = Math.max(1, Math.ceil(h / 2))
      this.levels.push(createStorageTexture(device, w, h))
    }
  }

  destroy(): void {
    this.levels.forEach((texture) => texture.destroy())
    this.levels = []
  }

  /** Records the reduction and returns the 1x1 result. */
  reduce(encoder: GPUCommandEncoder, source: GPUTexture): GPUTexture {
    let input = source

    for (const output of this.levels) {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: input.createView() },
          { binding: 1, resource: output.createView() },
        ],
      })

      const pass = encoder.beginComputePass()
      pass.setPipeline(this.pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(Math.ceil(output.width / 8), Math.ceil(output.height / 8))
      pass.end()

      input = output
    }

    return input
  }
}
