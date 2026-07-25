export interface WebGPUContext {
  device: GPUDevice
  context: GPUCanvasContext
  format: GPUTextureFormat
  /** True if `timestamp-query` was available and enabled, so GPU timings can be measured. */
  hasTimestamps: boolean
  /** Resizes the swapchain to the canvas's current CSS size. Returns false if unchanged. */
  resizeToDisplay: () => boolean
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<WebGPUContext> {
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No suitable GPUAdapter found.')

  // Optional: without it we fall back to wall-clock timing, which vsync pins to the refresh rate.
  const hasTimestamps = adapter.features.has('timestamp-query')
  const device = await adapter.requestDevice({
    requiredFeatures: hasTimestamps ? ['timestamp-query'] : [],
  })

  const context = canvas.getContext('webgpu')
  if (!context) throw new Error('Failed to get WebGPU canvas context.')

  const format = navigator.gpu.getPreferredCanvasFormat()

  const resizeToDisplay = () => {
    const dpr = window.devicePixelRatio || 1
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr))
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr))
    if (canvas.width === width && canvas.height === height) return false

    canvas.width = width
    canvas.height = height
    context.configure({ device, format, alphaMode: 'opaque' })
    return true
  }
  resizeToDisplay()

  return { device, context, format, hasTimestamps, resizeToDisplay }
}
