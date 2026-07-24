import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import divergenceWGSL from '../shaders/divergence.wgsl?raw'
import testVelocityFieldsWGSL from '../shaders/testVelocityFields.wgsl?raw'

export class DivergenceDebug {
  readonly rotationDivergenceView: GPUTextureView
  readonly radialDivergenceView: GPUTextureView
  readonly rotationDivergenceTexture: GPUTexture
  readonly radialDivergenceTexture: GPUTexture

  constructor(device: GPUDevice, width: number, height: number) {
    const divergencePipeline = createComputePipeline(device, divergenceWGSL)
    const bakeRotationPipeline = createComputePipeline(device, testVelocityFieldsWGSL, 'rotation')
    const bakeRadialPipeline = createComputePipeline(device, testVelocityFieldsWGSL, 'radial')

    const rotationVelocity = createStorageTexture(device, width, height)
    const radialVelocity = createStorageTexture(device, width, height)
    const rotationDivergence = createStorageTexture(device, width, height)
    const radialDivergence = createStorageTexture(device, width, height)

    this.rotationDivergenceView = rotationDivergence.createView()
    this.radialDivergenceView = radialDivergence.createView()
    this.rotationDivergenceTexture = rotationDivergence
    this.radialDivergenceTexture = radialDivergence

    const workgroupsX = Math.ceil(width / 8)
    const workgroupsY = Math.ceil(height / 8)

    const encoder = device.createCommandEncoder()

    const dispatchBake = (pipeline: GPUComputePipeline, output: GPUTexture) => {
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: output.createView() }],
      })
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupsX, workgroupsY)
      pass.end()
    }

    const dispatchDivergence = (velocity: GPUTexture, output: GPUTexture) => {
      const bindGroup = device.createBindGroup({
        layout: divergencePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: velocity.createView() },
          { binding: 1, resource: output.createView() },
        ],
      })
      const pass = encoder.beginComputePass()
      pass.setPipeline(divergencePipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupsX, workgroupsY)
      pass.end()
    }

    dispatchBake(bakeRotationPipeline, rotationVelocity)
    dispatchBake(bakeRadialPipeline, radialVelocity)
    dispatchDivergence(rotationVelocity, rotationDivergence)
    dispatchDivergence(radialVelocity, radialDivergence)

    device.queue.submit([encoder.finish()])
  }
}
