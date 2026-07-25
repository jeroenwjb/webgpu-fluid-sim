@group(0) @binding(0) var velocityTex: texture_2d<f32>;   // vx in .r, vy in .g
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

const VIS_SCALE: f32 = 5.0; // tweak if the visualization is too faint/saturated

// FORWARD differences, deliberately paired with the BACKWARD differences in
// gradientSubtract.wgsl. Composing the two reproduces exactly the compact Laplacian
// (p[i+1] - 2p[i] + p[i-1]) that jacobi.wgsl inverts, so the pressure solve is consistent
// with the operators around it. Central differences (right - left) * 0.5 would skip the
// center texel and leave an irreducible divergence residual.
fn computeDivergence(coord: vec2i, dims: vec2i) -> f32 {
  let maxCoord = dims - vec2i(1, 1);

  let here = textureLoad(velocityTex, coord, 0);
  let right = textureLoad(velocityTex, clamp(coord + vec2i(1, 0), vec2i(0), maxCoord), 0).x;
  let up = textureLoad(velocityTex, clamp(coord + vec2i(0, 1), vec2i(0), maxCoord), 0).y;

  return (right - here.x) + (up - here.y);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(velocityTex));

  let divergence = computeDivergence(coord, dims);

  let positive = max(divergence, 0.0) * VIS_SCALE;
  let negative = max(-divergence, 0.0) * VIS_SCALE;
  let color = vec4f(positive, 0.0, negative, 1.0);

  textureStore(outputTex, coord, color);
}

// Raw (uncolored) divergence value, for feeding into the pressure solve.
@compute @workgroup_size(8, 8)
fn raw(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(velocityTex));

  let divergence = computeDivergence(coord, dims);

  textureStore(outputTex, coord, vec4f(divergence, 0.0, 0.0, 1.0));
}
