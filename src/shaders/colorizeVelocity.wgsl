@group(0) @binding(0) var sourceTex: texture_2d<f32>;  // vx in .r, vy in .g
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

const VIS_SCALE: f32 = 0.004;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let velocity = textureLoad(sourceTex, coord, 0).xy * VIS_SCALE;

  // Rightward/leftward flow reads red, up/down green, biased so still fluid is mid-grey.
  textureStore(outputTex, coord, vec4f(velocity * 0.5 + 0.5, 0.5, 1.0));
}
