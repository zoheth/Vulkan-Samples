#version 320 es

precision highp float;

layout(location = 0) in vec4 frag_pos;

layout(set = 0, binding = 6) uniform ShadowUniform
{
	vec3 light_position;
	float shadow_far_plane;
} shadow_uniform;

void main(void)
{
    float light_distance = length(frag_pos.xyz - shadow_uniform.light_position);

    // map to [0;1] range by dividing by far_plane
    light_distance = light_distance / shadow_uniform.shadow_far_plane;

    // write this as modified depth
    gl_FragDepth = light_distance;
}