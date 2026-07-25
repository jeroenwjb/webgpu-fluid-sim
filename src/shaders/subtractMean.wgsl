@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var meanTex: texture_2d<f32>;  // 1x1
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let mean = textureLoad(meanTex, vec2i(0, 0), 0).x;
  let value = textureLoad(sourceTex, coord, 0).x;

  textureStore(outputTex, coord, vec4f(value - mean, 0.0, 0.0, 1.0));
}
