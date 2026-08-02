/**
 * Binding shorthand for explicit layouts.
 *
 * `layout: 'auto'` always infers a filterable `float` sample type for `texture_2d<f32>`, which
 * r32float can't satisfy - it's `unfilterable-float`. So any pipeline that binds pressure or
 * divergence needs its layout spelled out. `unfilterable-float` accepts rgba16float too, so
 * the same layout works for both.
 */
export type BindingSpec =
  | 'uniform'
  | 'texture'
  | 'sampler'
  | { storage: GPUTextureFormat; access?: GPUStorageTextureAccess }

export function createBindGroupLayout(device: GPUDevice, specs: BindingSpec[]): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: specs.map((spec, binding) => {
      const entry: GPUBindGroupLayoutEntry = { binding, visibility: GPUShaderStage.COMPUTE }
      if (spec === 'uniform') entry.buffer = { type: 'uniform' }
      else if (spec === 'texture') entry.texture = { sampleType: 'unfilterable-float' }
      else if (spec === 'sampler') entry.sampler = { type: 'filtering' }
      else entry.storageTexture = { format: spec.storage, access: spec.access ?? 'write-only' }
      return entry
    }),
  })
}

/** Pressure and divergence are scalar, so they use a single channel and support read_write. */
export const SCALAR_FORMAT: GPUTextureFormat = 'r32float'

/**
 * jacobi.wgsl is shared with diffusion, which needs all four channels of rgba16float. Patching
 * the storage format keeps one source file rather than a near-duplicate that can drift.
 */
export function asScalarShader(code: string): string {
  return code.replaceAll('rgba16float', SCALAR_FORMAT)
}

export function createComputePipeline(
  device: GPUDevice,
  code: string,
  entryPoint = 'main',
  bindings?: BindingSpec[],
): GPUComputePipeline {
  const module = device.createShaderModule({ code })
  const layout = bindings
    ? device.createPipelineLayout({ bindGroupLayouts: [createBindGroupLayout(device, bindings)] })
    : 'auto'
  return device.createComputePipeline({ layout, compute: { module, entryPoint } })
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
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
  })
}
