struct AdvectParams {
  dt: f32,
  angularSpeed: f32,
}

@group(0) @binding(0) var<uniform> params: AdvectParams;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var sourceTex: texture_2d<f32>;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(sourceTex));
  let pos = vec2f(coord);

  let texCenter = dims * 0.5;
  let offset = pos - texCenter;

  let velocity = params.angularSpeed * vec2f(offset.y , -offset.x);
  let previousPos = pos - velocity * params.dt;
  let previousUV = previousPos / dims;
  let value = textureSampleLevel(sourceTex, linearSampler, previousUV, 0.0);
  
  textureStore(outputTex, coord, value);
}
