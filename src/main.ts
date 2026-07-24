import './style.css'
import { initWebGPU } from './webgpu/context'
import { createRenderPipeline } from './webgpu/renderPipeline'
import { createComputePipeline } from './webgpu/computePipeline'
import { Field } from './sim/field'
import incrementWGSL from './shaders/increment.wgsl?raw'

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
  const incrementPipeline = createComputePipeline(device, incrementWGSL)

  const field = new Field(device, SIM_WIDTH, SIM_HEIGHT)
  const sampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })

  function frame() {
    const encoder = device.createCommandEncoder()

    const computeBindGroup = device.createBindGroup({
      layout: incrementPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: field.read.createView() },
        { binding: 1, resource: field.write.createView() },
      ],
    })
    const computePass = encoder.beginComputePass()
    computePass.setPipeline(incrementPipeline)
    computePass.setBindGroup(0, computeBindGroup)
    computePass.dispatchWorkgroups(Math.ceil(SIM_WIDTH / 8), Math.ceil(SIM_HEIGHT / 8))
    computePass.end()

    field.swap()

    const renderBindGroup = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: field.read.createView() },
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
