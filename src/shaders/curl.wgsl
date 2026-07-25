@group(0) @binding(0) var velocityTex: texture_2d<f32>;  // vx in .r, vy in .g
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(velocityTex));

  let maxCoord = dims - vec2i(1, 1);

  // Cross pairs, subtracted - divergence uses the matching pairs, added.
  let left = textureLoad(velocityTex, clamp(coord + vec2i(-1, 0), vec2i(0), maxCoord), 0).y;
  let right = textureLoad(velocityTex, clamp(coord + vec2i(1, 0), vec2i(0), maxCoord), 0).y;
  let down = textureLoad(velocityTex, clamp(coord + vec2i(0, -1), vec2i(0), maxCoord), 0).x;
  let up = textureLoad(velocityTex, clamp(coord + vec2i(0, 1), vec2i(0), maxCoord), 0).x;

  let curl = ((right - left) - (up - down)) * 0.5;

  textureStore(outputTex, coord, vec4f(curl, 0.0, 0.0, 1.0));
}
