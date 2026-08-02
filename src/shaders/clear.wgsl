@group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(outputTex));
  if (coord.x >= dims.x || coord.y >= dims.y) {
    return;
  }
  textureStore(outputTex, coord, vec4f(0.0));
}
