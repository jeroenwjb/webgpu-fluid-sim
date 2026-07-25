@group(0) @binding(0) var sourceTex: texture_2d<f32>;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba16float, write>;

// abs() only matters on the first level; later levels are already positive.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let outDims = vec2i(textureDimensions(outputTex));
  if (coord.x >= outDims.x || coord.y >= outDims.y) {
    return;
  }

  let maxCoord = vec2i(textureDimensions(sourceTex)) - vec2i(1, 1);
  let base = coord * 2;

  let a = abs(textureLoad(sourceTex, clamp(base, vec2i(0), maxCoord), 0).x);
  let b = abs(textureLoad(sourceTex, clamp(base + vec2i(1, 0), vec2i(0), maxCoord), 0).x);
  let c = abs(textureLoad(sourceTex, clamp(base + vec2i(0, 1), vec2i(0), maxCoord), 0).x);
  let d = abs(textureLoad(sourceTex, clamp(base + vec2i(1, 1), vec2i(0), maxCoord), 0).x);

  textureStore(outputTex, coord, vec4f((a + b + c + d) * 0.25, 0.0, 0.0, 1.0));
}
