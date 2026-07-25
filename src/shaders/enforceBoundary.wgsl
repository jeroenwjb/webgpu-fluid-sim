@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

const MARGIN: f32 = 4.0;   // ramp width, avoids a hard edge
const DECAY: f32 = 0.9995; // guards against slow energy build-up

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let dims = vec2f(textureDimensions(sourceTex));
  var value = textureLoad(sourceTex, coord, 0) * DECAY;

  let distLeft = f32(coord.x);
  let distRight = dims.x - 1.0 - f32(coord.x);
  let distTop = f32(coord.y);
  let distBottom = dims.y - 1.0 - f32(coord.y);

  // Free-slip: damp only the component pointing through the wall, leave flow along it alone.
  let horizontalWall = max(1.0 - smoothstep(0.0, MARGIN, distLeft), 1.0 - smoothstep(0.0, MARGIN, distRight));
  let verticalWall = max(1.0 - smoothstep(0.0, MARGIN, distTop), 1.0 - smoothstep(0.0, MARGIN, distBottom));

  value.x *= 1.0 - horizontalWall;
  value.y *= 1.0 - verticalWall;

  textureStore(outputTex, coord, value);
}
