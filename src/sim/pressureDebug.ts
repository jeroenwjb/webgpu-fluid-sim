import { createComputePipeline, createStorageTexture } from '../webgpu/computePipeline'
import divergenceWGSL from '../shaders/divergence.wgsl?raw'
import jacobiWGSL from '../shaders/jacobi.wgsl?raw'
import colorizeWGSL from '../shaders/colorizeSigned.wgsl?raw'
import testVelocityFieldsWGSL from '../shaders/testVelocityFields.wgsl?raw'
import { Field } from './field'

const PRESSURE_ITERATIONS = 40

export class PressureDebug {
  readonly pressureView: GPUTextureView
  readonly pressureTexture: GPUTexture

  constructor(device: GPUDevice, width: number, height: number) {
    const bakeRadialPipeline = createComputePipeline(device, testVelocityFieldsWGSL, 'radial')
    const divergenceRawPipeline = createComputePipeline(device, divergenceWGSL, 'raw')
    const jacobiPipeline = createComputePipeline(device, jacobiWGSL)
    const colorizePipeline = createComputePipeline(device, colorizeWGSL)

    const radialVelocity = createStorageTexture(device, width, height)
    const divergenceTex = createStorageTexture(device, width, height)
    const pressureField = new Field(device, width, height)
    const pressureVis = createStorageTexture(device, width, height)

    this.pressureView = pressureVis.createView()
    this.pressureTexture = pressureVis

    const workgroupsX = Math.ceil(width / 8)
    const workgroupsY = Math.ceil(height / 8)

    // Poisson pressure equation: p = (divergence + sum_of_neighbors(p)) / 4 -> alpha=1, rBeta=1/4.
    const uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([1, 0.25]))

    const encoder = device.createCommandEncoder()

    const dispatch = (
      pipeline: GPUComputePipeline,
      entries: { binding: number; resource: GPUBindingResource }[],
    ) => {
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
      const pass = encoder.beginComputePass()
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(workgroupsX, workgroupsY)
      pass.end()
    }

    dispatch(bakeRadialPipeline, [{ binding: 0, resource: radialVelocity.createView() }])

    dispatch(divergenceRawPipeline, [
      { binding: 0, resource: radialVelocity.createView() },
      { binding: 1, resource: divergenceTex.createView() },
    ])

    // p starts at zero (freshly created Field), divergence is the fixed source term.
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
      dispatch(jacobiPipeline, [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: pressureField.read.createView() },
        { binding: 2, resource: divergenceTex.createView() },
        { binding: 3, resource: pressureField.write.createView() },
      ])
      pressureField.swap()
    }

    dispatch(colorizePipeline, [
      { binding: 0, resource: pressureField.read.createView() },
      { binding: 1, resource: pressureVis.createView() },
    ])

    device.queue.submit([encoder.finish()])
  }
}
