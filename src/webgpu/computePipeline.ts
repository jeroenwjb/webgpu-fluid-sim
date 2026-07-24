export function createComputePipeline(
  device: GPUDevice,
  code: string,
  entryPoint = 'main',
): GPUComputePipeline {
  const module = device.createShaderModule({ code })
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint },
  })
}

export function createStorageTexture(
  device: GPUDevice,
  width: number,
  height: number,
  format: GPUTextureFormat = 'rgba16float',
): GPUTexture {
  return device.createTexture({
    size: { width, height },
    format,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  })
}
