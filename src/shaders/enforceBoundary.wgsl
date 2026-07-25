@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

const MARGIN: f32 = 4.0;   // texels over which the wall condition ramps in, avoids a hard edge
const DECAY: f32 = 0.9995; // tiny per-frame dissipation, guards against residual drift

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(sourceTex));
  var value = textureLoad(sourceTex, coord, 0) * DECAY;

  let distLeft = f32(coord.x);
  let distRight = dims.x - 1.0 - f32(coord.x);
  let distTop = f32(coord.y);
  let distBottom = dims.y - 1.0 - f32(coord.y);

  // Free-slip (no-penetration) walls: damp only the component pointing THROUGH the wall,
  // leaving flow along the wall intact. Unlike reflecting the component (v -> -v), this
  // removes energy at the boundary instead of injecting it, so the projection feedback
  // loop stays stable.
  let horizontalWall = max(1.0 - smoothstep(0.0, MARGIN, distLeft), 1.0 - smoothstep(0.0, MARGIN, distRight));
  let verticalWall = max(1.0 - smoothstep(0.0, MARGIN, distTop), 1.0 - smoothstep(0.0, MARGIN, distBottom));

  value.x *= 1.0 - horizontalWall;
  value.y *= 1.0 - verticalWall;

  textureStore(outputTex, coord, value);
}
