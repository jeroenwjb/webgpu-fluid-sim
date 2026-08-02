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
| `S` | switch pressure solver |
| `P` | toggle the performance overlay |
| `B` | run a benchmark (results in the console) |

## Performance

The overlay shows per-stage GPU timings from WebGPU timestamp queries. Wall-clock frame time
is useless here — `requestAnimationFrame` is vsync-capped, so it reads a flat 60/100/144 fps
regardless of load and tells you nothing about headroom.

Projection dominates the frame, because it runs 60 solver iterations where every other stage
runs one. So it's the only part worth optimising, and the sim ships with three solvers you can
switch between at runtime and benchmark against each other. It runs on multigrid by default;
`S` steps down to red-black Gauss-Seidel and then Jacobi.

`B` runs a scripted stroke — 180 warm-up frames then 600 measured — with input derived purely
from the frame index so every solver sees identical strokes, and timings resolved GPU-side with
no readback until the run ends. Two things it took to make the numbers reproducible:

- **Don't pace a GPU benchmark on `requestAnimationFrame`.** At ~3 ms of work per 16.7 ms frame
  the GPU idles most of the time and never holds a boost clock. Identical runs varied 6×, and
  `min` came in at a sixth of the median. Submitting back-to-back via
  `queue.onSubmittedWorkDone()` keeps it saturated and collapsed the spread to ~5%. A longer
  warm-up did *not* help — it was never a settling problem.
- **Compare medians.** p95 runs 3–5× median from occasional stalls, so means swing ~20% between
  runs while medians hold to ~5%.

On an RTX 4070:

| solver | `project` | residual |
|---|---|---|
| Jacobi, 60 iterations | 0.70 ms | 1.35 |
| red-black Gauss-Seidel, 60 sweeps | 1.18 ms | 0.72 |
| **multigrid, 1 V-cycle** | **0.59 ms** | **0.019** |

Residual is the RMS of `Laplacian(p) − divergence` — how far the solve actually got. Timing
alone would be misleading: a solver that's fast because it converges worse isn't faster.

Multigrid is ~70× more accurate than Jacobi *and* cheaper. That isn't a better smoother — it's
a different strategy. Jacobi and Gauss-Seidel both kill high-frequency error quickly and crawl
on smooth error, which is most of what a pressure field contains. Multigrid solves the smooth
part on coarse grids, where it no longer looks smooth, and carries the correction back. Only 4
smoothing sweeps run at full resolution; everything below is on quarter-size grids and smaller,
so the whole hierarchy costs less than one extra full-res pass.

**You can see the difference without reading the numbers.** Dye is advected by the velocity
field, so leftover divergence means places where more dye arrives than leaves. Splatting is
additive, so it piles up there and blows out to white. Switch to Jacobi and the plumes wash out
noticeably faster than on multigrid.

Three things I got wrong along the way, all caught by measuring:

- **"Gauss-Seidel converges twice as fast" is asymptotic.** At 60 iterations on a 512 grid,
  ρ_Jacobi ≈ 0.99998 and ρ_GS ≈ 0.99996 — raised to those powers they're indistinguishable,
  because neither has touched the low-frequency error. My first red-black was a *wash* with
  Jacobi, and that's why.
- **Ping-pong wasted the ordering.** Every sweep had to write the whole grid and copy the
  untouched colour through, costing 3× for 2× the convergence. Updating in place dropped it
  from 4.48 ms to 1.18 ms.
- **Multigrid needs a grid-spacing scale.** The stencil is the unscaled `Σn − 4p` with no
  `/h²` — self-consistent on one grid, wrong across levels, since on doubled spacing the same
  stencil returns 4× as much. Without a ×4 on the restricted residual, corrections came back
  four times too small and multigrid scored *worse* than Jacobi. One constant fixed it.

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
- **Scalar fields are `r32float`.** Pressure and divergence are single-channel, so three of
  four `rgba16float` channels were wasted — and `read_write` storage textures, which in-place
  red-black needs, are only allowed for the 32-bit single-channel formats. Two catches follow
  from that: `layout: 'auto'` always infers a filterable `float` sample type while `r32float`
  is `unfilterable-float`, so every pipeline binding them needs an explicit layout; and
  nothing can bilinearly sample them, so multigrid's prolongation does its own interpolation
  with four `textureLoad`s rather than using a sampler.
- **Coarse grids solve for the error, not the pressure.** This is the part of multigrid that
  isn't obvious. You can't usefully transfer a solution between grids, but if `p̃` is the
  current approximation then the error `e = p − p̃` satisfies `∇²e = b − ∇²p̃ = residual` —
  the same operator, and `e` is smooth after smoothing, which is exactly what a coarse grid
  handles well. So each level down solves for a correction to the level above it.
- **Pressure needs its mean removed.** With free-slip walls everywhere the Poisson problem is
  pure Neumann, so pressure is only defined up to a constant — and since it's warm-started
  from the previous frame, nothing pins that constant down and it random-walks. Invisible to
  the flow, which only ever uses the gradient, but it would eventually saturate the texture.
  The divergence needs its mean removed too, for a different reason: the discrete problem is
  only solvable if the source sums to zero, and one-sided stencils clamping on opposite edges
  leave it slightly off.
- **Texel centres are at `(i + 0.5) / dims`.** Advection originally traced back to
  `previousPos / dims`, half a texel off, which shifted the whole field ~0.5 texels *per
  advection step* — twice a frame at 60 fps. It presented as the entire sim slowly crawling
  toward one corner, and I spent a while blaming the discretisation before checking the
  coordinate convention. The same trap sits in multigrid's prolongation, where a misaligned
  correction doesn't look broken, it just quietly stops converging.
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
  changed — most parameters are constant, so the per-frame CPU→GPU traffic is near zero. This
  came out of a bug rather than foresight: `queue.writeBuffer()` lands at *submit* time, not
  when the encoder records, so a pass class reusing one uniform buffer across two calls in the
  same frame had the second call's parameters clobber the first. The dye field was silently
  getting the velocity splat's parameters, which rendered everything green.
- **Sim resolution is independent of display resolution.** The grid tracks the canvas aspect
  ratio so the fluid never stretches, and is reallocated at frame start on resize —
  mid-encode would invalidate bind groups.

## Structure

```
src/
  webgpu/      device + canvas setup, pipeline/layout helpers, GPU profiler, resource pools
  sim/         one class per simulation stage
    solvers/   the three pressure solvers behind a common interface
  shaders/     one .wgsl per pass
  input/       pointer tracking, kept WebGPU-unaware
```

Adding a solver means implementing `PressureSolver` and registering it — it's then switchable
with `S` and benchmarkable with `B` without touching the projection code.

## Running locally

```bash
npm install
npm run dev
```

Build: `npm run build`. TypeScript, Vite, no framework.

## Reference

- Jos Stam, [*Stable Fluids*](https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf) (1999)
- Mark Harris, [*Fast Fluid Dynamics Simulation on the GPU*](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu), GPU Gems ch. 38
