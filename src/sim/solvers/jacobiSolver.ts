import { createComputePipeline, SCALAR_FORMAT, asScalarShader } from '../../webgpu/computePipeline'
import { dispatch } from '../../webgpu/dispatch'
import jacobiWGSL from '../../shaders/jacobi.wgsl?raw'
import type { PressureSolver } from './pressureSolver'
import type { Field } from '../field'

/**
 * Plain Jacobi relaxation: every new value comes from the neighbours' old values, so each
 * iteration is one dispatch and the whole grid updates in parallel.
 *
 * Simple and the baseline the other solvers are measured against, but slow to converge -
 * low-frequency error decays at only ~(1 - O(1/N^2)) per iteration.
 */
export class JacobiSolver implements PressureSolver {
  readonly name = 'jacobi'
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private uniformBuffer: GPUBuffer
  private iterations: number

  constructor(device: GPUDevice, iterations: number) {
    this.device = device
    this.iterations = iterations
    this.pipeline = createComputePipeline(device, asScalarShader(jacobiWGSL), 'main', [
      'uniform',
      'texture',
      'texture',
      { storage: SCALAR_FORMAT },
    ])

    // Laplacian(p) = div  ->  p = (neighbours - div) / 4
    this.uniformBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([-1, 0.25]))
  }

  solve(
    encoder: GPUCommandEncoder,
    pressure: Field,
    divergence: GPUTexture,
    width: number,
    height: number,
  ): void {
    for (let i = 0; i < this.iterations; i++) {
      dispatch(
        this.device,
        encoder,
        this.pipeline,
        [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: pressure.read.createView() },
          { binding: 2, resource: divergence.createView() },
          { binding: 3, resource: pressure.write.createView() },
        ],
        width,
        height,
      )
      pressure.swap()
    }
  }

  destroy(): void {
    this.uniformBuffer.destroy()
  }
}
