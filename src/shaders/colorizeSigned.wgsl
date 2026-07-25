@group(0) @binding(0) var sourceTex: texture_2d<f32>;  // signed scalar in .r
@group(0) @binding(1) var scaleTex: texture_2d<f32>;   // 1x1, mean |value| in the field
@group(0) @binding(2) var outputTex: texture_storage_2d<rgba16float, write>;

// Black at zero so quiet regions stay dark, brightness for magnitude, hue for sign.
const COOL = vec3f(0.25, 0.55, 1.0);
const WARM = vec3f(1.0, 0.35, 0.2);
const MID = vec3f(0.0, 0.0, 0.0);

// Multiplier on the mean, so typical values land mid-ramp instead of saturating.
const GAIN: f32 = 3.0;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let coord = id.xy;
  let value = textureLoad(sourceTex, coord, 0).x;

  // Auto-scale to whatever is on screen; a fixed scale either clips or shows nothing,
  // since magnitudes swing wildly with how hard the fluid is being pushed.
  let scale = max(textureLoad(scaleTex, vec2i(0, 0), 0).x, 1e-6) * GAIN;
  let t = clamp(value / scale, -1.0, 1.0);

  // sqrt curve: these fields are mostly small values with rare spikes, so a linear ramp
  // leaves the interesting structure nearly black.
  let shaped = sign(t) * sqrt(abs(t));
  let color = select(mix(MID, COOL, -shaped), mix(MID, WARM, shaped), shaped > 0.0);

  textureStore(outputTex, coord, vec4f(color, 1.0));
}
