@group(0) @binding(0) var coarseTex: texture_2d<f32>;
// Read-modify-write: the correction is added to what's already here, not substituted for it.
@group(0) @binding(1) var fineTex: texture_storage_2d<rgba16float, read_write>;

// Bilinear by hand rather than with a sampler: r32float is unfilterable-float, so no filtering
// sampler may bind it, and the coarse grid has to be r32float because the smoother writes it
// in place (read_write only exists for the 32-bit single-channel formats).
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let fineDims = vec2i(textureDimensions(fineTex));
  if (coord.x >= fineDims.x || coord.y >= fineDims.y) {
    return;
  }

  let coarseDims = vec2i(textureDimensions(coarseTex));
  let scale = vec2f(coarseDims) / vec2f(fineDims);

  // Both grids cover the same domain, so match texel CENTRES: fine centre (i + 0.5) maps to
  // coarse index (i + 0.5) * scale - 0.5. Dropping either half-texel shifts the correction and
  // quietly wrecks convergence.
  let coarsePos = (vec2f(coord) + 0.5) * scale - 0.5;
  let base = vec2i(floor(coarsePos));
  let f = coarsePos - floor(coarsePos);

  let maxCoord = coarseDims - vec2i(1, 1);
  let c00 = textureLoad(coarseTex, clamp(base, vec2i(0), maxCoord), 0);
  let c10 = textureLoad(coarseTex, clamp(base + vec2i(1, 0), vec2i(0), maxCoord), 0);
  let c01 = textureLoad(coarseTex, clamp(base + vec2i(0, 1), vec2i(0), maxCoord), 0);
  let c11 = textureLoad(coarseTex, clamp(base + vec2i(1, 1), vec2i(0), maxCoord), 0);

  let correction = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);

  // The fine grid already holds a partly-solved answer; the coarse grid only computed what
  // was missing from it.
  textureStore(fineTex, coord, textureLoad(fineTex, coord) + correction);
}
