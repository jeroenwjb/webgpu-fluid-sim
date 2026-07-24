@group(0) @binding(0) var velocityTex: texture_2d<f32>;   // vx in .r, vy in .g
@group(0) @binding(1) var pressureTex: texture_2d<f32>;   // p in .r
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(velocityTex));

  let pLeft  = textureLoad(pressureTex, clamp(coord + vec2i(-1, 0), vec2i(0), dims - vec2i(1, 1)), 0).x;
  let pRight  = textureLoad(pressureTex, clamp(coord + vec2i(1, 0), vec2i(0), dims - vec2i(1, 1)), 0).x;

  let pUp  = textureLoad(pressureTex, clamp(coord + vec2i(0, 1), vec2i(0), dims - vec2i(1, 1)), 0).x;
  let pDown  = textureLoad(pressureTex, clamp(coord + vec2i(0, -1), vec2i(0), dims - vec2i(1, 1)), 0).x;

  let gradient = vec2f(pRight - pLeft, pUp - pDown) * 0.5;
  let sourceVel = textureLoad(velocityTex, coord, 0).xy;

  let newVel = sourceVel - gradient;
  textureStore(outputTex, coord, vec4f(newVel, 0.0, 1.0));


}
