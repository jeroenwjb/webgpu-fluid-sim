import './style.css'
import { initWebGPU } from './webgpu/context'
import { createRenderPipeline } from './webgpu/renderPipeline'
import { Field } from './sim/field'
import { SplatPass } from './sim/splat'
import { AdvectPass } from './sim/advect'
import { DiffusionPass } from './sim/diffusion'
import { DivergenceDebug } from './sim/divergenceDebug'

const SIM_WIDTH = 512
const SIM_HEIGHT = 512
const DT = 1 / 60
const ANGULAR_SPEED = 1.5

// alpha/rBeta for the Jacobi diffusion solve: alpha = dx^2 / (viscosity * dt), rBeta = 1 / (alpha + 4).
const DIFFUSION_VISCOSITY = 8
const DIFFUSION_ALPHA = 1 / (DIFFUSION_VISCOSITY * DT)
const DIFFUSION_RBETA = 1 / (DIFFUSION_ALPHA + 4)
const DIFFUSION_ITERATIONS = 25

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

  const field = new Field(device, SIM_WIDTH, SIM_HEIGHT)
  const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })

  let debugMode: 1 | 2 | 3 = 1
  window.addEventListener('keydown', (e) => {
    if (e.key === '1' || e.key === '2' || e.key === '3') {
      debugMode = Number(e.key) as 1 | 2 | 3
    }
  })

  // Splat is additive, so we only apply it once into an otherwise-static field.
  // Placed off-center so the rotation (velocity is zero at the center) is visible.
  {
    const encoder = device.createCommandEncoder()
    splatPass.apply(encoder, field, SIM_WIDTH, SIM_HEIGHT, {
      x: SIM_WIDTH * 0.7,
      y: SIM_HEIGHT * 0.5,
      radius: 60,
      strength: 1,
      color: [0.9, 0.3, 0.1, 1],
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
    advectPass.apply(encoder, field, SIM_WIDTH, SIM_HEIGHT, { dt: DT, angularSpeed: ANGULAR_SPEED })

    const displayView =
      debugMode === 1
        ? field.read.createView()
        : debugMode === 2
          ? divergenceDebug.rotationDivergenceView
          : divergenceDebug.radialDivergenceView

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
