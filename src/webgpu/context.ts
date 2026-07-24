export interface WebGPUContext {
  device: GPUDevice
  context: GPUCanvasContext
  format: GPUTextureFormat
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<WebGPUContext> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No suitable GPUAdapter found.')

  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu')
  if (!context) throw new Error('Failed to get WebGPU canvas context.')

  const format = navigator.gpu.getPreferredCanvasFormat()

  const configure = () => {
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr))
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr))
    context.configure({ device, format, alphaMode: 'opaque' })
  }
  configure()

  return { device, context, format }
}
