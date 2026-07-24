@group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn rotation(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(outputTex));
  let offset = vec2f(coord) - dims * 0.5;
  let velocity = vec2f(-offset.y, offset.x);
  textureStore(outputTex, coord, vec4f(velocity, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn radial(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(outputTex));
  let offset = vec2f(coord) - dims * 0.5;
  // Gaussian falloff so the outward push decays to ~zero at the boundary
  // (net-zero flux out of the domain -> a well-posed test for zero-flux boundaries).
  let radius = min(dims.x, dims.y) * 0.2;
  let falloff = exp(-dot(offset, offset) / (radius * radius));
  let velocity = offset * falloff;
  textureStore(outputTex, coord, vec4f(velocity, 0.0, 1.0));
}
