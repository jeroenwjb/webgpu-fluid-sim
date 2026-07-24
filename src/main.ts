import './style.css'
import { initWebGPU } from './webgpu/context'
import { createRenderPipeline } from './webgpu/renderPipeline'
import { createComputePipeline, createStorageTexture } from './webgpu/computePipeline'
import fillWGSL from './shaders/fill.wgsl?raw'

const SIM_WIDTH = 512
const SIM_HEIGHT = 512

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
  const computePipeline = createComputePipeline(device, fillWGSL)

  const patternTexture = createStorageTexture(device, SIM_WIDTH, SIM_HEIGHT)
  const patternTextureView = patternTexture.createView()

  const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })

  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: patternTextureView }],
  })

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: patternTextureView },
      { binding: 1, resource: sampler },
    ],
  })

  // Pattern is static, so we only need to run the compute pass once.
  {
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(computePipeline)
    pass.setBindGroup(0, computeBindGroup)
    pass.dispatchWorkgroups(Math.ceil(SIM_WIDTH / 8), Math.ceil(SIM_HEIGHT / 8))
    pass.end()
    device.queue.submit([encoder.finish()])
  }

  function frame() {
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: 'clear',
          clearValue: { r: 0.02, g: 0.12, b: 0.16, a: 1 },
          storeOp: 'store',
        },
      ],
    })
    pass.setPipeline(renderPipeline)
    pass.setBindGroup(0, renderBindGroup)
    pass.draw(3)
    pass.end()
    device.queue.submit([encoder.finish()])
    requestAnimationFrame(frame)
  }

  requestAnimationFrame(frame)
}

main()
