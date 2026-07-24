@group(0) @binding(0) var readTex: texture_2d<f32>;
@group(0) @binding(1) var writeTex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let coord = id.xy;

    let color = textureLoad(readTex, coord, 0);
    let incrementedColor = vec4f(color.x + 0.01, color.y, color.z, 1.0);
    let clampedColor = clamp(incrementedColor, vec4f(0.0), vec4f(1.0));
    textureStore(writeTex, coord, clampedColor);
}