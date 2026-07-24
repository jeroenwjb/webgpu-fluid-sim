@group(0) @binding(0) var outputTex: texture_storage_2d<rgba16float, write>;

const TILE_SIZE: u32 = 32u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;

  let cellX = coord.x / TILE_SIZE;
  let cellY = coord.y / TILE_SIZE;
  let isEven = (cellX + cellY) % 2u == 0u;
  let color = select(vec4f(1.0, 0.0, 0.0, 1.0), vec4f(0.0, 1.0, 0.0, 1.0), isEven);

  textureStore(outputTex, coord, color);
}
