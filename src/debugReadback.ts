// Reads rgba16float texels back to the console, for checking fields numerically.

function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1
  const exponent = (bits >> 10) & 0x1f
  const fraction = bits & 0x3ff
  if (exponent === 0) return sign * 6.103515625e-5 * (fraction / 1024)
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024)
}

export async function logTextureRow(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  y: number,
  label: string,
): Promise<void> {
  const bytesPerRow = width * 8 // rgba16float = 4 channels * 2 bytes
  const buffer = device.createBuffer({
    size: bytesPerRow,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const encoder = device.createCommandEncoder()
  encoder.copyTextureToBuffer(
    { texture, origin: { x: 0, y, z: 0 } },
    { buffer, bytesPerRow },
    { width, height: 1 },
  )
  device.queue.submit([encoder.finish()])

  await buffer.mapAsync(GPUMapMode.READ)
  const bits = new Uint16Array(buffer.getMappedRange())

  const samples: number[] = []
  for (let x = 0; x < width; x += 32) {
    samples.push(Math.round(decodeFloat16(bits[x * 4]) * 1000) / 1000) // red channel, every 32nd texel
  }
  buffer.unmap()

  console.log(`[${label}] red channel samples (every 32px):`, samples)
}

export async function logTexelRGBA(
  device: GPUDevice,
  texture: GPUTexture,
  x: number,
  y: number,
  label: string,
): Promise<void> {
  const bytesPerRow = 256 // WebGPU requires bytesPerRow to be a multiple of 256
  const buffer = device.createBuffer({
    size: bytesPerRow,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  })

  const encoder = device.createCommandEncoder()
  encoder.copyTextureToBuffer({ texture, origin: { x, y, z: 0 } }, { buffer, bytesPerRow }, { width: 1, height: 1 })
  device.queue.submit([encoder.finish()])

  await buffer.mapAsync(GPUMapMode.READ)
  const bits = new Uint16Array(buffer.getMappedRange())
  const rgba = [
    decodeFloat16(bits[0]),
    decodeFloat16(bits[1]),
    decodeFloat16(bits[2]),
    decodeFloat16(bits[3]),
  ].map((v) => Math.round(v * 1000) / 1000)
  buffer.unmap()

  console.log(`[${label}] RGBA at (${x},${y}):`, rgba)
}
