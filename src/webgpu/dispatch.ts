/** Builds a bind group and runs one compute pass over a width x height grid. */
export function dispatch(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  entries: { binding: number; resource: GPUBindingResource }[],
  width: number,
  height: number,
): void {
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })

  const pass = encoder.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8))
  pass.end()
}
