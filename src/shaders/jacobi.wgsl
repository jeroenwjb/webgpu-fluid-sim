struct JacobiParams {
  alpha: f32,
  rBeta: f32,
}

@group(0) @binding(0) var<uniform> params: JacobiParams;
@group(0) @binding(1) var xTex: texture_2d<f32>;   // current best-guess field (ping-pongs each iteration)
@group(0) @binding(2) var bTex: texture_2d<f32>;   // fixed source term (stays constant across iterations)
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(xTex));

  // TODO:
  let left  = textureLoad(xTex, clamp(coord + vec2i(-1, 0), vec2i(0), dims - vec2i(1, 1)), 0);
  let right  = textureLoad(xTex, clamp(coord + vec2i(1, 0), vec2i(0), dims - vec2i(1, 1)), 0);
  let up  = textureLoad(xTex, clamp(coord + vec2i(0, 1), vec2i(0), dims - vec2i(1, 1)), 0);
  let down  = textureLoad(xTex, clamp(coord + vec2i(0, -1), vec2i(0), dims - vec2i(1, 1)), 0);
  // - Load xTex at coord's 4 grid neighbors (left/right/up/down, i.e. coord +/- 1
  //   in each axis). Clamp each neighbor coordinate into [0, dims-1] first, so
  //   border texels don't read outside the texture.
  // - Load bTex at this coord (the fixed source value here).
  let source = textureLoad(bTex, coord, 0);

  let newValue = (left +right+up+down+params.alpha*source)*params.rBeta;
  // - Combine: the generic Jacobi update is
  //     newValue = (sum of the 4 neighbor values + alpha * source) * rBeta
  // - Write newValue to outputTex.
  textureStore(outputTex, coord, newValue);
  
}
