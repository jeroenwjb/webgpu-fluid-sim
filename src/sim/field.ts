export class Field {
  private textureA: GPUTexture
  private textureB: GPUTexture
  private readIsA = true

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    format: GPUTextureFormat = 'rgba16float',
  ) {
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC
    this.textureA = device.createTexture({ size: { width, height }, format, usage })
    this.textureB = device.createTexture({ size: { width, height }, format, usage })
  }

  get read(): GPUTexture {
    return this.readIsA ? this.textureA : this.textureB
  }

  get write(): GPUTexture {
    return this.readIsA ? this.textureB : this.textureA
  }

  swap(): void {
    this.readIsA = !this.readIsA
  }
}
