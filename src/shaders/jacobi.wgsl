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

  let left  = textureLoad(xTex, clamp(coord + vec2i(-1, 0), vec2i(0), dims - vec2i(1, 1)), 0);
  let right  = textureLoad(xTex, clamp(coord + vec2i(1, 0), vec2i(0), dims - vec2i(1, 1)), 0);
  let up  = textureLoad(xTex, clamp(coord + vec2i(0, 1), vec2i(0), dims - vec2i(1, 1)), 0);
  let down  = textureLoad(xTex, clamp(coord + vec2i(0, -1), vec2i(0), dims - vec2i(1, 1)), 0);
  let source = textureLoad(bTex, coord, 0);

  let newValue = (left +right+up+down+params.alpha*source)*params.rBeta;
  textureStore(outputTex, coord, newValue);
  
}
