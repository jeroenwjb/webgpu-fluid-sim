import { createComputePipeline } from '../webgpu/computePipeline'
import enforceBoundaryWGSL from '../shaders/enforceBoundary.wgsl?raw'
import type { Field } from './field'

export class BoundaryPass {
  private pipeline: GPUComputePipeline
  private device: GPUDevice

  constructor(device: GPUDevice) {
    this.device = device
    this.pipeline = createComputePipeline(device, enforceBoundaryWGSL)
  }

  apply(encoder: GPUCommandEncoder, field: Field, width: number, height: number): void {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: field.read.createView() },
        { binding: 1, resource: field.write.createView() },
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
