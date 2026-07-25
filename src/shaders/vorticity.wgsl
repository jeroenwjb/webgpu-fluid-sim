struct VorticityParams {
  strength: f32,
  dt: f32,
}

@group(0) @binding(0) var<uniform> params: VorticityParams;
@group(0) @binding(1) var velocityTex: texture_2d<f32>;
@group(0) @binding(2) var curlTex: texture_2d<f32>;  // scalar curl in .r
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = vec2i(id.xy);
  let dims = vec2i(textureDimensions(velocityTex));

  // Gradient of |curl| points toward the centre of a vortex.
  let cLeft  = abs(textureLoad(curlTex, clamp(coord + vec2i(-1, 0), vec2i(0), dims - vec2i(1, 1)), 0).x);
  let cRight  = abs(textureLoad(curlTex, clamp(coord + vec2i(1, 0), vec2i(0), dims - vec2i(1, 1)), 0).x);
  let cUp  = abs(textureLoad(curlTex, clamp(coord + vec2i(0, 1), vec2i(0), dims - vec2i(1, 1)), 0).x);
  let cDown  = abs(textureLoad(curlTex, clamp(coord + vec2i(0, -1), vec2i(0), dims - vec2i(1, 1)), 0).x);
  let curl = textureLoad(curlTex, coord, 0).x;
  let velocity = textureLoad(velocityTex, coord, 0).xy;

  let gradient = vec2f(cRight - cLeft, cUp - cDown) * 0.5;
  let scaledGradient = gradient / max(length(gradient), 1e-5);

  // Perpendicular, signed by curl, so each vortex gets pushed the way it already spins.
  let force = vec2f(scaledGradient.y, -scaledGradient.x) * curl;

  let newVelocity = velocity + force * params.strength * params.dt;
  textureStore(outputTex, coord, vec4f(newVelocity, 0.0, 1.0));
}
