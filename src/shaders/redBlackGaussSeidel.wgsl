struct RedBlackParams {
  alpha: f32,
  rBeta: f32,
  parity: f32, // 0 updates one colour, 1 the other
}

@group(0) @binding(0) var<uniform> params: RedBlackParams;
@group(0) @binding(1) var bTex: texture_2d<f32>;  // divergence, fixed across iterations
// In place: read and written through the same binding, so there is no separate output.
// asScalarShader() rewrites the format to r32float when building the pipeline.
@group(0) @binding(2) var pressureTex: texture_storage_2d<rgba16float, read_write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(pressureTex));

  // Checkerboard colour. A 5-point stencil only ever reads the opposite colour, so updating
  // one at a time is race-free.
  let isEven = (coord.x + coord.y) % 2i == 0;
  let shouldUpdate = isEven == (params.parity > 0.5);

  // Wrong colour: the value is already correct, so skip the write entirely.
  if (!shouldUpdate) {
    return;
  }

  let left  = textureLoad(pressureTex, clamp(coord + vec2i(-1, 0), vec2i(0), dims - vec2i(1, 1)));
  let right  = textureLoad(pressureTex, clamp(coord + vec2i(1, 0), vec2i(0), dims - vec2i(1, 1)));
  let up  = textureLoad(pressureTex, clamp(coord + vec2i(0, 1), vec2i(0), dims - vec2i(1, 1)));
  let down  = textureLoad(pressureTex, clamp(coord + vec2i(0, -1), vec2i(0), dims - vec2i(1, 1)));
  let source = textureLoad(bTex, coord, 0);

  let newValue = (left + right + up + down + params.alpha * source) * params.rBeta;
  textureStore(pressureTex, coord, newValue);
}
