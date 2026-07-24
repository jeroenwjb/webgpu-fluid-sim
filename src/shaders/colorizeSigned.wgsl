@group(0) @binding(0) var sourceTex: texture_2d<f32>;   // signed scalar value in .r
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

const VIS_SCALE: f32 = 5.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let value = textureLoad(sourceTex, coord, 0).x;

  let positive = max(value, 0.0) * VIS_SCALE;
  let negative = max(-value, 0.0) * VIS_SCALE;

  textureStore(outputTex, coord, vec4f(positive, 0.0, negative, 1.0));
}
