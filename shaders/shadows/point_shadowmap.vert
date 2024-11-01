#version 320 es

layout(location = 0) in vec3 position;

layout(set = 0, binding = 1) uniform GlobalUniform {
    mat4 model;
    mat4 view_proj;
    vec3 camera_position;
} global_uniform;

layout(location = 0) out vec4 frag_pos;

void main(void)
{
    frag_pos= global_uniform.model * vec4(position, 1.0);
    gl_Position = global_uniform.view_proj * frag_pos;
}