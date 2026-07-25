@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let outDims = vec2i(textureDimensions(outputTex));
  if (coord.x >= outDims.x || coord.y >= outDims.y) {
    return;
  }

  let maxCoord = vec2i(textureDimensions(sourceTex)) - vec2i(1, 1);
  let base = coord * 2;

  let a = textureLoad(sourceTex, clamp(base, vec2i(0), maxCoord), 0);
  let b = textureLoad(sourceTex, clamp(base + vec2i(1, 0), vec2i(0), maxCoord), 0);
  let c = textureLoad(sourceTex, clamp(base + vec2i(0, 1), vec2i(0), maxCoord), 0);
  let d = textureLoad(sourceTex, clamp(base + vec2i(1, 1), vec2i(0), maxCoord), 0);

  textureStore(outputTex, coord, (a + b + c + d) * 0.25);
}
