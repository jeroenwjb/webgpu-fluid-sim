struct AdvectParams {
  dt: f32,
  dissipation: f32, // 1.0 for velocity; below 1.0 fades dye so strokes don't pile up to white
}

@group(0) @binding(0) var<uniform> params: AdvectParams;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var velocityTex: texture_2d<f32>;  // drives the trace
@group(0) @binding(3) var sourceTex: texture_2d<f32>;    // field being carried, may be velocityTex
@group(0) @binding(4) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(sourceTex));
  let pos = vec2f(coord);

  let velocity = textureLoad(velocityTex, coord, 0).xy;
  let previousPos = pos - velocity * params.dt;
  let previousUV = (previousPos + 0.5) / dims; // texel centres sit at (i + 0.5) / dims
  let value = textureSampleLevel(sourceTex, linearSampler, previousUV, 0.0);

  textureStore(outputTex, coord, value * params.dissipation);
}
