struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    const positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );

    let pos = positions[vertexIndex];

    var out: VertexOutput;
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = (pos + 1.0) / 2.0;
    out.uv.y = 1.0 - out.uv.y;
    return out;
}