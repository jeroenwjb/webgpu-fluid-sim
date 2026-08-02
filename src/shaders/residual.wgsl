@group(0) @binding(0) var pressureTex: texture_2d<f32>;
@group(0) @binding(1) var divergenceTex: texture_2d<f32>;
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

// How far the current pressure is from actually solving Laplacian(p) = divergence.
// .r keeps the signed residual for the debug view, .g its square so an averaging reduction
// yields the mean square in one pass.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let maxCoord = vec2i(textureDimensions(pressureTex)) - vec2i(1, 1);

  let left = textureLoad(pressureTex, clamp(coord + vec2i(-1, 0), vec2i(0), maxCoord), 0).x;
  let right = textureLoad(pressureTex, clamp(coord + vec2i(1, 0), vec2i(0), maxCoord), 0).x;
  let up = textureLoad(pressureTex, clamp(coord + vec2i(0, 1), vec2i(0), maxCoord), 0).x;
  let down = textureLoad(pressureTex, clamp(coord + vec2i(0, -1), vec2i(0), maxCoord), 0).x;
  let here = textureLoad(pressureTex, coord, 0).x;

  let laplacian = left + right + up + down - 4.0 * here;
  let residual = textureLoad(divergenceTex, coord, 0).x - laplacian;

  textureStore(outputTex, coord, vec4f(residual, residual * residual, 0.0, 1.0));
}
