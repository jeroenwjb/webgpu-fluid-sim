import { createComputePipeline, SCALAR_FORMAT, asScalarShader } from '../../webgpu/computePipeline'
import { dispatch } from '../../webgpu/dispatch'
import redBlackWGSL from '../../shaders/redBlackGaussSeidel.wgsl?raw'
import type { PressureSolver } from './pressureSolver'
import type { Field } from '../field'

/**
 * Gauss-Seidel with checkerboard ordering, updating in place.
 *
 * A 5-point stencil only ever reads the opposite colour, so one colour can be updated at a
 * time with no races - and because nothing reads what this pass writes, it can write straight
 * back into the same texture. That removes the ping-pong version's copy-through, which cost a
 * full-grid write and an extra read per sweep.
 */
export class RedBlackSolver implements PressureSolver {
  readonly name = 'red-black GS'
  private device: GPUDevice
  private pipeline: GPUComputePipeline
  private redBuffer: GPUBuffer
  private blackBuffer: GPUBuffer
  private iterations: number

  constructor(device: GPUDevice, iterations: number) {
    this.device = device
    this.iterations = iterations
    this.pipeline = createComputePipeline(device, asScalarShader(redBlackWGSL), 'main', [
      'uniform',
      'texture',
      { storage: SCALAR_FORMAT, access: 'read-write' },
    ])

    // Two buffers rather than one rewritten per pass: queue.writeBuffer lands at submit time,
    // so a shared buffer would give every pass whichever parity was written last.
    const makeParams = (parity: number) => {
      const buffer = device.createBuffer({
        size: 16, // alpha, rBeta, parity + pad
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      device.queue.writeBuffer(buffer, 0, new Float32Array([-1, 0.25, parity, 0]))
      return buffer
    }
    this.redBuffer = makeParams(0)
    this.blackBuffer = makeParams(1)
  }

  solve(
    encoder: GPUCommandEncoder,
    pressure: Field,
    divergence: GPUTexture,
    width: number,
    height: number,
  ): void {
    // No swap: each pass writes back into the same texture, so pressure.read stays current
    // throughout and the black pass sees the reds this sweep just updated.
    for (let i = 0; i < this.iterations; i++) {
      for (const params of [this.redBuffer, this.blackBuffer]) {
        dispatch(
          this.device,
          encoder,
          this.pipeline,
          [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: divergence.createView() },
            { binding: 2, resource: pressure.read.createView() },
          ],
          width,
          height,
        )
      }
    }
  }

  destroy(): void {
    this.redBuffer.destroy()
    this.blackBuffer.destroy()
  }
}
