struct Splat {
  position: vec2f,
  radius: f32,
  strength: f32,
  color: vec4f,
}

@group(0) @binding(0) var<uniform> splat: Splat;
@group(0) @binding(1) var readTex: texture_2d<f32>;
@group(0) @binding(2) var writeTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  
  let coord = id.xy;
  let color = textureLoad(readTex, coord, 0);
  let dist = distance(vec2f(coord), splat.position);
  let falloff = exp(-dist*dist / (splat.radius*splat.radius));
  let newColor = clamp(color + falloff *splat.strength * splat.color, vec4f(0), vec4f(1));

  textureStore(writeTex, coord, newColor);
}
