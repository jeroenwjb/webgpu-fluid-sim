@group(0) @binding(0) var fineTex: texture_2d<f32>;
@group(0) @binding(1) var coarseTex: texture_storage_2d<rgba16float, write>;

// The stencil is the unscaled Sn - 4p, with no /h^2. That's fine on one grid, but across
// levels it isn't: for smooth data the same stencil on doubled spacing returns 4x as much,
// since L ~ h^2 * laplacian. Without this the coarse grid solves L(e) = r/4 and the correction
// comes back four times too small.
const LEVEL_SCALE: f32 = 4.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let coarseDims = vec2i(textureDimensions(coarseTex));
  if (coord.x >= coarseDims.x || coord.y >= coarseDims.y) {
    return;
  }

  // Each coarse cell averages the 2x2 fine block beneath it, which damps leftover
  // high-frequency error rather than aliasing it onto the coarse grid.
  let maxCoord = vec2i(textureDimensions(fineTex)) - vec2i(1, 1);
  let base = coord * 2;

  let a = textureLoad(fineTex, clamp(base, vec2i(0), maxCoord), 0);
  let b = textureLoad(fineTex, clamp(base + vec2i(1, 0), vec2i(0), maxCoord), 0);
  let c = textureLoad(fineTex, clamp(base + vec2i(0, 1), vec2i(0), maxCoord), 0);
  let d = textureLoad(fineTex, clamp(base + vec2i(1, 1), vec2i(0), maxCoord), 0);

  textureStore(coarseTex, coord, (a + b + c + d) * 0.25 * LEVEL_SCALE);
}
