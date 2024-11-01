#version 320 es

precision highp float;
precision highp samplerCubeShadow;

#ifdef HAS_BASE_COLOR_TEXTURE
layout(set = 0, binding = 0) uniform sampler2D base_color_texture;
#endif

layout(location = 0) in vec4 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec3 in_normal;

layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 1) uniform GlobalUniform
{
	mat4 model;
	mat4 view_proj;
	vec3 camera_position;
}
global_uniform;

// Push constants come with a limitation in the size of data.
// The standard requires at least 128 bytes
layout(push_constant, std430) uniform PBRMaterialUniform
{
	vec4  base_color_factor;
	float metallic_factor;
	float roughness_factor;
}
pbr_material_uniform;

#include "lighting.h"

layout(set = 0, binding = 4) uniform LightsInfo
{
	Light directional_lights[MAX_LIGHT_COUNT];
	Light point_lights[MAX_LIGHT_COUNT];
	Light spot_lights[MAX_LIGHT_COUNT];
}
lights_info;

layout(constant_id = 1) const uint POINT_LIGHT_COUNT = 0U;

layout(set = 0, binding = 5) uniform samplerCubeShadow shadowmap_texture;

layout(set = 0, binding = 6) uniform ShadowUniform
{
	vec3 light_position;
	float shadow_far_plane;
}
shadow_uniform;


float calculate_point_light_shadow(vec3 frag_pos)
{
    vec3 frag_to_light = frag_pos - shadow_uniform.light_position;

    float current_depth = length(frag_to_light);

    return texture(shadowmap_texture, vec4(frag_to_light, current_depth / shadow_uniform.shadow_far_plane));
}

void main(void)
{
	vec3 normal = normalize(in_normal);

	vec3 light_contribution = vec3(0.0);

//	o_color = vec4(vec3(calculate_point_light_shadow(in_pos.xyz)), 1.0);
//	return;

	for (uint i = 0U; i < POINT_LIGHT_COUNT; ++i)
	{
		Light point_light = lights_info.point_lights[i];
		vec3 light_dir = point_light.position.xyz - in_pos.xyz;

		light_contribution += apply_point_light(point_light, in_pos.xyz, normal);

		if (i == 0U) 
        {
            light_contribution *= calculate_point_light_shadow(in_pos.xyz);
        }
	}

	vec4 base_color = vec4(1.0, 0.0, 0.0, 1.0);

#ifdef HAS_BASE_COLOR_TEXTURE
	base_color = texture(base_color_texture, in_uv);
#else
	base_color = pbr_material_uniform.base_color_factor;
#endif

	vec3 ambient_color = vec3(0.2) * base_color.xyz;

	o_color = vec4(ambient_color + light_contribution * base_color.xyz, base_color.w);
}