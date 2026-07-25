import './style.css'
import { initWebGPU } from './webgpu/context'
import { createRenderPipeline } from './webgpu/renderPipeline'
import { Field } from './sim/field'
import { SplatPass } from './sim/splat'
import { AdvectPass } from './sim/advect'
import { DiffusionPass } from './sim/diffusion'
import { DebugView, DEBUG_FIELDS } from './sim/debugView'
import { Legend } from './legend'
import { ProjectionPass } from './sim/projection'
import { BoundaryPass } from './sim/boundary'
import { VorticityPass } from './sim/vorticity'
import { PointerTracker } from './input/pointer'
import { Stats } from './stats'
import { GpuProfiler } from './webgpu/gpuProfiler'

// Other axis follows the canvas aspect so the fluid never stretches. Independent of display res.
const SIM_MAX_AXIS = 512
const DT = 1 / 60
const PROJECTION_ITERATIONS = 60

const SPLAT_RADIUS = 25
const VELOCITY_FORCE = 6 // scaled by drag speed
const DYE_AMOUNT = 0.12

// Splatting is additive, so dye needs a fade or it saturates to white.
const DYE_DISSIPATION = 0.995
const VELOCITY_DISSIPATION = 1

// alpha = 1 / (viscosity * dt), rBeta = 1 / (alpha + 4).
// Careful: bigger alpha means less smoothing, not more.
const DIFFUSION_VISCOSITY = 0.5
const DIFFUSION_ALPHA = 1 / (DIFFUSION_VISCOSITY * DT)
const DIFFUSION_RBETA = 1 / (DIFFUSION_ALPHA + 4)
const DIFFUSION_ITERATIONS = 5

// Puts back the small vortices semi-Lagrangian advection smears away. Too high and it adds
// energy faster than diffusion removes it.
const VORTICITY_STRENGTH = 12

// Damps checkerboard noise rather than modelling viscosity, so it runs stronger than the dye's.
const VELOCITY_DIFFUSION_VISCOSITY = 2
const VELOCITY_DIFFUSION_ALPHA = 1 / (VELOCITY_DIFFUSION_VISCOSITY * DT)
const VELOCITY_DIFFUSION_RBETA = 1 / (VELOCITY_DIFFUSION_ALPHA + 4)
const VELOCITY_DIFFUSION_ITERATIONS = 20

const canvas = document.querySelector<HTMLCanvasElement>('#fluid-canvas')!
const fallback = document.querySelector<HTMLDivElement>('#webgpu-fallback')!
const statsElement = document.querySelector<HTMLDivElement>('#stats')!
const legend = new Legend(
  document.querySelector<HTMLDivElement>('#legend')!,
  document.querySelector<HTMLSpanElement>('#legend-min')!,
  document.querySelector<HTMLSpanElement>('#legend-max')!,
)

/** Rounded to the 8x8 workgroup size. */
function simSizeForCanvas(): { width: number; height: number } {
  const aspect = Math.max(canvas.clientWidth, 1) / Math.max(canvas.clientHeight, 1)
  const width = aspect >= 1 ? SIM_MAX_AXIS : SIM_MAX_AXIS * aspect
  const height = aspect >= 1 ? SIM_MAX_AXIS / aspect : SIM_MAX_AXIS
  const round8 = (v: number) => Math.max(8, Math.round(v / 8) * 8)
  return { width: round8(width), height: round8(height) }
}

/** Fully saturated hue -> RGB, so strokes cycle through vivid colours. */
function hueToRgb(h: number): [number, number, number] {
  const f = (n: number) => {
    const k = (n + h * 6) % 6
    return Math.max(0, Math.min(1, Math.min(k, 4 - k, 1)))
  }
  return [f(5), f(3), f(1)]
}

function showFallback() {
  canvas.hidden = true
  fallback.hidden = false
}

async function main() {
  if (!navigator.gpu) {
    showFallback()
    return
  }

  let device: GPUDevice
  let context: GPUCanvasContext
  let format: GPUTextureFormat
  let resizeToDisplay: () => boolean
  let hasTimestamps: boolean
  try {
    ;({ device, context, format, hasTimestamps, resizeToDisplay } = await initWebGPU(canvas))
  } catch (err) {
    console.error('WebGPU init failed:', err)
    showFallback()
    return
  }

  const renderPipeline = createRenderPipeline(device, format)
  const splatPass = new SplatPass(device)
  const advectPass = new AdvectPass(device)
  const boundaryPass = new BoundaryPass(device)
  const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })

  let { width: simWidth, height: simHeight } = simSizeForCanvas()

  // Rebuilt on resize.
  let debugView = new DebugView(device, simWidth, simHeight)
  let diffusionPass = new DiffusionPass(device, simWidth, simHeight)
  let projectionPass = new ProjectionPass(device, simWidth, simHeight)
  let vorticityPass = new VorticityPass(device, simWidth, simHeight)
  let field = new Field(device, simWidth, simHeight)
  let velocityField = new Field(device, simWidth, simHeight)

  const stats = new Stats(statsElement)
  // 'total' has to be first - Stats treats it as the frame total.
  const profiler = hasTimestamps
    ? new GpuProfiler(device, [
        'total',
        'splat',
        'diffuse dye',
        'diffuse vel',
        'advect vel',
        'vorticity',
        'project',
        'boundary',
        'advect dye',
        'render',
      ])
    : null
  const pointer = new PointerTracker(canvas, simWidth, simHeight)

  let debugField = DEBUG_FIELDS[0]
  const updateSimDetail = () => stats.setDetail(`sim ${simWidth}x${simHeight}   view: ${debugField}`)
  updateSimDetail()

  let resizePending = false
  window.addEventListener('resize', () => {
    resizePending = true
  })

  // Runs at frame start - reallocating mid-encode would invalidate bind groups.
  function applyResize() {
    resizePending = false
    if (!resizeToDisplay()) return

    const next = simSizeForCanvas()
    if (next.width === simWidth && next.height === simHeight) return

    simWidth = next.width
    simHeight = next.height

    // Dropped rather than resampled: the dye clears, but resizes are rare.
    diffusionPass.destroy()
    projectionPass.destroy()
    vorticityPass.destroy()
    debugView.destroy()
    debugView = new DebugView(device, simWidth, simHeight)
    diffusionPass = new DiffusionPass(device, simWidth, simHeight)
    projectionPass = new ProjectionPass(device, simWidth, simHeight)
    vorticityPass = new VorticityPass(device, simWidth, simHeight)
    field = new Field(device, simWidth, simHeight)
    velocityField = new Field(device, simWidth, simHeight)

    pointer.setSimSize(simWidth, simHeight)
    updateSimDetail()
  }

  window.addEventListener('keydown', (e) => {
    const index = Number(e.key) - 1
    if (index >= 0 && index < DEBUG_FIELDS.length) {
      debugField = DEBUG_FIELDS[index]
      updateSimDetail()
    }
    if (e.key.toLowerCase() === 'p') stats.toggle()
  })

  let hue = 0

  function frame() {
    if (resizePending) applyResize()
    stats.frame(profiler ? profiler.timings : [])

    const encoder = device.createCommandEncoder()
    profiler?.begin(encoder, 'total')

    // Add sources first (matches Stam's "add source before diffuse/advect" ordering).
    const input = pointer.consume()
    if (input.isDown && input.moved) {
      hue = (hue + 0.005) % 1
      const [r, g, b] = hueToRgb(hue)

      profiler?.begin(encoder, 'splat')
      splatPass.apply(encoder, velocityField, simWidth, simHeight, {
        x: input.x,
        y: input.y,
        radius: SPLAT_RADIUS,
        strength: VELOCITY_FORCE,
        color: [input.dx, input.dy, 0, 0],
      })
      splatPass.apply(encoder, field, simWidth, simHeight, {
        x: input.x,
        y: input.y,
        radius: SPLAT_RADIUS,
        strength: DYE_AMOUNT,
        color: [r, g, b, 1],
      })
      profiler?.end(encoder, 'splat')
    }

    profiler?.begin(encoder, 'diffuse dye')
    diffusionPass.apply(encoder, field, simWidth, simHeight, {
      alpha: DIFFUSION_ALPHA,
      rBeta: DIFFUSION_RBETA,
      iterations: DIFFUSION_ITERATIONS,
    })
    profiler?.end(encoder, 'diffuse dye')

    // Velocity step: diffuse, self-advect, project.
    profiler?.begin(encoder, 'diffuse vel')
    diffusionPass.apply(encoder, velocityField, simWidth, simHeight, {
      alpha: VELOCITY_DIFFUSION_ALPHA,
      rBeta: VELOCITY_DIFFUSION_RBETA,
      iterations: VELOCITY_DIFFUSION_ITERATIONS,
    })
    profiler?.end(encoder, 'diffuse vel')

    profiler?.begin(encoder, 'advect vel')
    advectPass.apply(encoder, velocityField, velocityField, simWidth, simHeight, {
      dt: DT,
      dissipation: VELOCITY_DISSIPATION,
    })
    profiler?.end(encoder, 'advect vel')

    // Before projection - the force adds divergence that projection then removes.
    profiler?.begin(encoder, 'vorticity')
    vorticityPass.apply(encoder, velocityField, simWidth, simHeight, {
      strength: VORTICITY_STRENGTH,
      dt: DT,
    })
    profiler?.end(encoder, 'vorticity')

    // Walls before projection as well as after: with free-slip everywhere the Poisson problem
    // only has a solution if total divergence sums to zero, which needs no flow through the
    // walls. Otherwise the solve grows a large-scale ramp instead of converging.
    profiler?.begin(encoder, 'boundary')
    boundaryPass.apply(encoder, velocityField, simWidth, simHeight)
    profiler?.end(encoder, 'boundary')

    profiler?.begin(encoder, 'project')
    projectionPass.apply(encoder, velocityField, simWidth, simHeight, { iterations: PROJECTION_ITERATIONS })
    profiler?.end(encoder, 'project')

    boundaryPass.apply(encoder, velocityField, simWidth, simHeight)

    // Density step: carry dye on the now divergence-free velocity.
    profiler?.begin(encoder, 'advect dye')
    advectPass.apply(encoder, velocityField, field, simWidth, simHeight, {
      dt: DT,
      dissipation: DYE_DISSIPATION,
    })
    profiler?.end(encoder, 'advect dye')

    const colorized =
      debugField === 'dye'
        ? null
        : debugView.colorize(
            encoder,
            debugField,
            {
              velocity: velocityField.read,
              divergence: projectionPass.divergenceTexture,
              pressure: projectionPass.pressureTexture,
              curl: vorticityPass.curl,
            },
            simWidth,
            simHeight,
          )
    const displayView = colorized ?? field.read.createView()
    legend.update(device, colorized ? debugView.scaleTexture : null)

    const renderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: displayView },
        { binding: 1, resource: sampler },
      ],
    })
    profiler?.begin(encoder, 'render')
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0.02, g: 0.12, b: 0.16, a: 1 },
          storeOp: 'store',
        },
      ],
    })
    renderPass.setPipeline(renderPipeline)
    renderPass.setBindGroup(0, renderBindGroup)
    renderPass.draw(3)
    renderPass.end()
    profiler?.end(encoder, 'render')

    profiler?.end(encoder, 'total')
    profiler?.resolve(encoder)
    device.queue.submit([encoder.finish()])
    // After submit - mapping a buffer an unsubmitted encoder references is a validation error.
    profiler?.readback()

    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

main()
