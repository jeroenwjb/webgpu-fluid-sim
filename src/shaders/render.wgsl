@fragment
fn main(@location(0) uv: vec2f) -> @location(0) vec4f {
  return vec4f(uv.x, uv.y, 0.0, 1.0);
}