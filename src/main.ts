import './style.css'
import { initWebGPU } from './webgpu/context'
import { createRenderPipeline } from './webgpu/renderPipeline'
import { Field } from './sim/field'
import { SplatPass } from './sim/splat'
import { AdvectPass } from './sim/advect'
import { DiffusionPass } from './sim/diffusion'
import { DivergenceDebug } from './sim/divergenceDebug'
import { PressureDebug } from './sim/pressureDebug'
import { ProjectionDebug } from './sim/projectionDebug'
import { ProjectionPass } from './sim/projection'
import { BoundaryPass } from './sim/boundary'

const SIM_WIDTH = 512
const SIM_HEIGHT = 512
const DT = 1 / 60
const PROJECTION_ITERATIONS = 20

// alpha/rBeta for the Jacobi diffusion solve: alpha = dx^2 / (viscosity * dt), rBeta = 1 / (alpha + 4).
// NOTE: larger alpha (= smaller viscosity) means LESS smoothing, since the update
// `(neighbors + alpha*source) * rBeta` weights the original value more heavily.
// Dye only needs gentle softening - too much and it dissolves within a second or two.
const DIFFUSION_VISCOSITY = 0.1
const DIFFUSION_ALPHA = 1 / (DIFFUSION_VISCOSITY * DT)
const DIFFUSION_RBETA = 1 / (DIFFUSION_ALPHA + 4)
const DIFFUSION_ITERATIONS = 20

// Velocity diffusion exists mainly to damp high-frequency "checkerboard" noise (a known
// artifact of the collocated-grid central-difference div/grad scheme) rather than to model
// real viscosity - it needs to be noticeably stronger than the dye's to suppress the stripes.
const VELOCITY_DIFFUSION_VISCOSITY = 2
const VELOCITY_DIFFUSION_ALPHA = 1 / (VELOCITY_DIFFUSION_VISCOSITY * DT)
const VELOCITY_DIFFUSION_RBETA = 1 / (VELOCITY_DIFFUSION_ALPHA + 4)
const VELOCITY_DIFFUSION_ITERATIONS = 20

const canvas = document.querySelector<HTMLCanvasElement>('#fluid-canvas')!
const fallback = document.querySelector<HTMLDivElement>('#webgpu-fallback')!

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
  try {
    ;({ device, context, format } = await initWebGPU(canvas))
  } catch (err) {
    console.error('WebGPU init failed:', err)
    showFallback()
    return
  }

  const renderPipeline = createRenderPipeline(device, format)
  const splatPass = new SplatPass(device)
  const advectPass = new AdvectPass(device)
  const diffusionPass = new DiffusionPass(device, SIM_WIDTH, SIM_HEIGHT)
  const divergenceDebug = new DivergenceDebug(device, SIM_WIDTH, SIM_HEIGHT)
  const pressureDebug = new PressureDebug(device, SIM_WIDTH, SIM_HEIGHT)
  const projectionDebug = new ProjectionDebug(device, SIM_WIDTH, SIM_HEIGHT)
  const projectionPass = new ProjectionPass(device, SIM_WIDTH, SIM_HEIGHT)
  const boundaryPass = new BoundaryPass(device)

  const field = new Field(device, SIM_WIDTH, SIM_HEIGHT)
  const velocityField = new Field(device, SIM_WIDTH, SIM_HEIGHT)
  const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })

  let debugMode: 1 | 2 | 3 | 4 | 5 = 1
  window.addEventListener('keydown', (e) => {
    if (e.key === '1' || e.key === '2' || e.key === '3' || e.key === '4' || e.key === '5') {
      debugMode = Number(e.key) as 1 | 2 | 3 | 4 | 5
    }
  })

  // Seed the dye and velocity fields once with off-center splats (both additive, so a single
  // application each is enough). The velocity splat gives the self-advecting field an initial
  // push to evolve from.
  {
    const encoder = device.createCommandEncoder()
    splatPass.apply(encoder, field, SIM_WIDTH, SIM_HEIGHT, {
      x: SIM_WIDTH * 0.7,
      y: SIM_HEIGHT * 0.5,
      radius: 60,
      strength: 1,
      color: [0.9, 0.3, 0.1, 1],
    })
    splatPass.apply(encoder, velocityField, SIM_WIDTH, SIM_HEIGHT, {
      x: SIM_WIDTH * 0.7,
      y: SIM_HEIGHT * 0.5,
      radius: 80,
      strength: 1,
      color: [-90, 75, 0, 0],
    })
    device.queue.submit([encoder.finish()])
  }

  function frame() {
    const encoder = device.createCommandEncoder()

    diffusionPass.apply(encoder, field, SIM_WIDTH, SIM_HEIGHT, {
      alpha: DIFFUSION_ALPHA,
      rBeta: DIFFUSION_RBETA,
      iterations: DIFFUSION_ITERATIONS,
    })

    // Velocity step: diffuse (damps checkerboard noise), self-advect, then project to remove divergence.
    diffusionPass.apply(encoder, velocityField, SIM_WIDTH, SIM_HEIGHT, {
      alpha: VELOCITY_DIFFUSION_ALPHA,
      rBeta: VELOCITY_DIFFUSION_RBETA,
      iterations: VELOCITY_DIFFUSION_ITERATIONS,
    })
    advectPass.apply(encoder, velocityField, velocityField, SIM_WIDTH, SIM_HEIGHT, { dt: DT })
    projectionPass.apply(encoder, velocityField, SIM_WIDTH, SIM_HEIGHT, { iterations: PROJECTION_ITERATIONS })
    boundaryPass.apply(encoder, velocityField, SIM_WIDTH, SIM_HEIGHT)

    // Density step: advect dye by the real (now divergence-free) velocity field.
    advectPass.apply(encoder, velocityField, field, SIM_WIDTH, SIM_HEIGHT, { dt: DT })

    const displayView =
      debugMode === 1
        ? field.read.createView()
        : debugMode === 2
          ? divergenceDebug.rotationDivergenceView
          : debugMode === 3
            ? divergenceDebug.radialDivergenceView
            : debugMode === 4
              ? pressureDebug.pressureView
              : projectionDebug.divergenceAfterView

    const renderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: displayView },
        { binding: 1, resource: sampler },
      ],
    })
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

    device.queue.submit([encoder.finish()])
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

main()
