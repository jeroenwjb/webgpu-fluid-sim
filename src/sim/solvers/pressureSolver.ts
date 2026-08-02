import type { Field } from '../field'

/**
 * Solves Laplacian(p) = divergence for pressure.
 *
 * ProjectionPass owns the pressure Field and passes it in, so switching solvers at runtime
 * keeps the warm-started pressure instead of resetting it mid-flow.
 */
export interface PressureSolver {
  /** Shown in the stats overlay. */
  readonly name: string

  solve(
    encoder: GPUCommandEncoder,
    pressure: Field,
    divergence: GPUTexture,
    width: number,
    height: number,
  ): void

  destroy(): void
}
