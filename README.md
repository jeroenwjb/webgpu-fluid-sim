# WebGPU Fluid Simulation

A real-time 2D fluid simulation running in the browser on WebGPU, implementing Jos Stam's
*Stable Fluids* method. Drag to push the fluid around.

**[Live demo](https://jeroenwjb.github.io/webgpu-fluid-sim/)** — needs a WebGPU-capable browser
(Chrome/Edge 113+, or Firefox 141+).

![A dye plume billowing and shedding filaments](docs/screenshot.png)

## What it does

Every frame the simulation runs the full Stable Fluids pipeline as WebGPU compute passes:

1. **Splat** — inject velocity and dye where the pointer is, scaled by drag speed
2. **Diffuse** — Jacobi relaxation on dye and velocity
3. **Advect** — semi-Lagrangian: trace backwards along the velocity field and resample
4. **Vorticity confinement** — measure the curl and push energy back into it
5. **Project** — make the velocity divergence-free: compute divergence, solve a Poisson
   equation for pressure, subtract its gradient
6. **Render** — sample the dye field onto a fullscreen triangle

The projection step is what makes it look like a fluid rather than a smear. Advecting velocity
by itself introduces divergence — fluid appearing and disappearing — and subtracting the
pressure gradient removes it.

## Controls

| Key | |
|---|---|
| drag | inject dye and velocity |
| `1`–`5` | show dye / velocity / divergence / pressure / curl |
| `P` | toggle the performance overlay |

## Performance

The overlay shows per-stage GPU timings from WebGPU timestamp queries. Wall-clock frame time
is useless here — `requestAnimationFrame` is vsync-capped, so it reads a flat 60/100/144 fps
regardless of load and tells you nothing about headroom.

At 512×336 on an RTX 4070 the frame costs ~3.4 ms, of which:

| stage | ms | share |
|---|---|---|
| project | 2.59 | 76% |
| diffuse velocity | 0.50 | 15% |
| everything else | ~0.35 | 9% |

Projection dominates because it runs 60 Jacobi iterations where every other stage runs one.
Jacobi is also the weakest solver available — its low-frequency error decays at roughly
`1 - O(1/N²)` per iteration, so throwing more iterations at it hits diminishing returns fast.
Replacing it with red-black Gauss-Seidel and multigrid is the next thing I want to build.

## Implementation notes

- **Ping-pong textures.** No pass reads and writes the same texture in one dispatch; each one
  reads `field.read`, writes `field.write`, then swaps. This is the whole reason the classic
  race bug never shows up.
- **One Jacobi kernel.** Diffusion and the pressure solve are the same relaxation with
  different `alpha`/`rBeta`, so `jacobi.wgsl` is written once instead of drifting into two
  near-identical shaders.
- **Consistent difference stencils.** Divergence uses forward differences and the pressure
  gradient uses backward ones, so composing them reproduces exactly the compact Laplacian that
  the Jacobi solve inverts. With central differences on both, projection solves a subtly
  different problem than the one being measured and leaves divergence it can never remove —
  visible as blocky faceting in the dye.
- **Vorticity confinement.** Semi-Lagrangian advection is unconditionally stable but pays for
  it with numerical dissipation — every backward trace is a bilinear sample, which is a slight
  blur, so small vortices smear away within a frame or two. Confinement measures where rotation
  is concentrated and pushes energy back along it. It doesn't invent detail; it stops the
  numerics from erasing detail the simulation was already producing. It's also the difference
  between the plume above and a smooth coloured blob.
- **Free-slip walls.** Only the velocity component pointing through a wall is damped. Actually
  reflecting it (`v → -v`) injects energy into the projection feedback loop and the field
  diverges within seconds.
- **Pooled GPU resources.** Uniform buffers and scratch textures come from small rings rather
  than being allocated per pass, and uniform writes are skipped when the contents haven't
  changed — most parameters are constant, so the per-frame CPU→GPU traffic is near zero.

## Running locally

```bash
npm install
npm run dev
```

Build: `npm run build`.

## Reference

- Jos Stam, [*Stable Fluids*](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf) (1999)
- Mark Harris, [*Fast Fluid Dynamics Simulation on the GPU*](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu), GPU Gems ch. 38
