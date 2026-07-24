import fullscreenQuadWGSL from '../shaders/fullscreenQuad.wgsl?raw'
import renderWGSL from '../shaders/render.wgsl?raw'

export function createRenderPipeline(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  const vertexModule = device.createShaderModule({ code: fullscreenQuadWGSL })
  const fragmentModule = device.createShaderModule({ code: renderWGSL })

  return device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: vertexModule,
      entryPoint: 'main',
    },
    fragment: {
      module: fragmentModule,
      entryPoint: 'main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  })
}
