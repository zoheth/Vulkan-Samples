/* Copyright (c) 2019-2024, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "point_light_shadow.h"

#include "common/vk_common.h"
#include "filesystem/legacy.h"
#include "gltf_loader.h"
#include "gui.h"
#include "platform/platform.h"
#include "rendering/subpasses/forward_subpass.h"
#include "stats/stats.h"

PointShadowSubpass::PointShadowSubpass(vkb::RenderContext &render_context, vkb::ShaderSource &&vertex_source, vkb::ShaderSource &&fragment_source, vkb::sg::Scene &scene, vkb::sg::Camera &camera) :
    GeometrySubpass(render_context, std::move(vertex_source), std::move(fragment_source), scene, camera)
{
	std::array<glm::vec3, 6> directions = {
	    glm::vec3(1.0f, 0.0f, 0.0f),         // 右 +X
	    glm::vec3(-1.0f, 0.0f, 0.0f),        // 左 -X
	    glm::vec3(0.0f, 1.0f, 0.0f),         // 上 +Y
	    glm::vec3(0.0f, -1.0f, 0.0f),        // 下 -Y
	    glm::vec3(0.0f, 0.0f, 1.0f),         // 前 +Z
	    glm::vec3(0.0f, 0.0f, -1.0f)         // 后 -Z
	};

	std::array<glm::vec3, 6> up_vectors = {
	    glm::vec3(0.0f, -1.0f, 0.0f),        // 右
	    glm::vec3(0.0f, -1.0f, 0.0f),        // 左
	    glm::vec3(0.0f, 0.0f, 1.0f),         // 上
	    glm::vec3(0.0f, 0.0f, -1.0f),        // 下
	    glm::vec3(0.0f, -1.0f, 0.0f),        // 前
	    glm::vec3(0.0f, -1.0f, 0.0f)         // 后
	};

	for (size_t i = 0; i < 6; ++i)
	{
		view_matrices[i] = glm::lookAt(camera.get_node()->get_transform().get_translation(),
		                               camera.get_node()->get_transform().get_translation() + directions[i],
		                               up_vectors[i]);
	}
}

void PointShadowSubpass::draw(vkb::CommandBuffer &command_buffer)
{
	for (current_face = 0; current_face < 6; ++current_face)
	{
		GeometrySubpass::draw(command_buffer);
	}
}

void PointShadowSubpass::update_uniform(vkb::CommandBuffer &command_buffer, vkb::sg::Node &node, size_t thread_index)
{
	vkb::GlobalUniform global_uniform;

	global_uniform.camera_view_proj = camera.get_pre_rotation() * vkb::rendering::vulkan_style_projection(camera.get_projection()) * view_matrices[current_face];

	auto &render_frame = get_render_context().get_active_frame();
	auto &transform    = node.get_transform();

	auto allocation = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(vkb::GlobalUniform), thread_index);

	global_uniform.model = transform.get_world_matrix();

	global_uniform.camera_position = camera.get_node()->get_transform().get_translation();

	allocation.update(global_uniform);
	command_buffer.bind_buffer(allocation.get_buffer(), allocation.get_offset(), allocation.get_size(), 0, 1, 0);
}

void PointShadowSubpass::prepare_pipeline_state(vkb::CommandBuffer &command_buffer, VkFrontFace front_face, bool double_sided_material)
{
	vkb::RasterizationState rasterization_state{};
	rasterization_state.front_face        = front_face;
	rasterization_state.depth_bias_enable = VK_TRUE;

	if (double_sided_material)
	{
		rasterization_state.cull_mode = VK_CULL_MODE_NONE;
	}

	command_buffer.set_rasterization_state(rasterization_state);
	command_buffer.set_depth_bias(-1.4f, 0.0f, -1.7f);

	vkb::MultisampleState multisample_state{};
	multisample_state.rasterization_samples = get_sample_count();
	command_buffer.set_multisample_state(multisample_state);
}

vkb::PipelineLayout &PointShadowSubpass::prepare_pipeline_layout(vkb::CommandBuffer &command_buffer, const std::vector<vkb::ShaderModule *> &shader_modules)
{
	return GeometrySubpass::prepare_pipeline_layout(command_buffer, shader_modules);
}

void PointShadowSubpass::prepare_push_constants(vkb::CommandBuffer &command_buffer, vkb::sg::SubMesh &sub_mesh)
{
	GeometrySubpass::prepare_push_constants(command_buffer, sub_mesh);
}

PointMainSubpass::PointMainSubpass(vkb::RenderContext &render_context, vkb::ShaderSource &&vertex_source, vkb::ShaderSource &&fragment_source, vkb::sg::Scene &scene, vkb::sg::Camera &camera, vkb::sg::PerspectiveCamera &shadowmap_camera, std::vector<std::unique_ptr<vkb::core::ImageView>> &shadow_cube_views) :
	vkb::ForwardSubpass{render_context, std::move(vertex_source), std::move(fragment_source), scene, camera},
	shadowmap_camera_{shadowmap_camera},
	shadow_cube_views_{shadow_cube_views}
{
	shadow_uniform_.light_position   = shadowmap_camera_.get_node()->get_transform().get_translation();
	shadow_uniform_.shadow_far_plane = shadowmap_camera.get_far_plane();
}

void PointMainSubpass::prepare()
{
	ForwardSubpass::prepare();

	// Calculate valid filter
	VkFilter filter = VK_FILTER_LINEAR;
	vkb::make_filters_valid(get_render_context().get_device().get_gpu().get_handle(),
	                        vkb::get_suitable_depth_format(get_render_context().get_device().get_gpu().get_handle()), &filter);

	VkSamplerCreateInfo cubemap_shadowmap_sampler_create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
	cubemap_shadowmap_sampler_create_info.minFilter     = filter;
	cubemap_shadowmap_sampler_create_info.magFilter     = filter;
	cubemap_shadowmap_sampler_create_info.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	cubemap_shadowmap_sampler_create_info.compareEnable = VK_TRUE;
	cubemap_shadowmap_sampler_create_info.compareOp     = VK_COMPARE_OP_GREATER_OR_EQUAL;
	cubemap_shadowmap_sampler_create_info.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	cubemap_shadowmap_sampler_create_info.maxLod        = 1.0f;
	cubemap_shadowmap_sampler_create_info.minLod        = 0.0f;

	cubemap_shadowmap_sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	cubemap_shadowmap_sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	cubemap_shadowmap_sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	cubemap_shadowmap_sampler_ = std::make_unique<vkb::core::Sampler>(get_render_context().get_device(), cubemap_shadowmap_sampler_create_info);
}

void PointMainSubpass::draw(vkb::CommandBuffer &command_buffer)
{
	command_buffer.bind_image(*shadow_cube_views_[get_render_context().get_active_frame_index()], *cubemap_shadowmap_sampler_, 0, 5, 0);
	auto                 &render_frame  = get_render_context().get_active_frame();
	vkb::BufferAllocation shadow_buffer = render_frame.allocate_buffer(VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sizeof(PointShadowUniform));
	shadow_buffer.update(shadow_uniform_);
	// Bind the shadowmap uniform to the proper set nd binding in shader
	command_buffer.bind_buffer(shadow_buffer.get_buffer(), shadow_buffer.get_offset(), shadow_buffer.get_size(), 0, 6, 0);

	ForwardSubpass::draw(command_buffer);
}

PointLightShadow::PointLightShadow() = default;

bool PointLightShadow::prepare(const vkb::ApplicationOptions &options)
{
	if (!VulkanSample::prepare(options))
	{
		return false;
	}

	create_shadow_render_target(SHADOWMAP_RESOLUTION);

	// Load a scene from the assets folder
	load_scene("scenes/sponza/Sponza01.gltf");

	// Attach a move script to the camera component in the scene
	auto &camera_node = vkb::add_free_camera(get_scene(), "main_camera", get_render_context().get_surface_extent());
	camera            = &camera_node.get_component<vkb::sg::Camera>();

	auto light_pos   = glm::vec3(50.0f, 128.0f, -225.0f);
	auto light_color = glm::vec3(1.0, 1.0, 1.0);

	vkb::sg::LightProperties light_properties{};
	light_properties.color     = light_color;
	light_properties.intensity = 1.0f;

	auto &light = vkb::add_point_light(get_scene(), light_pos, light_properties);

	// Attach a camera component to the light node
	auto shadowmap_camera_ptr = std::make_unique<vkb::sg::PerspectiveCamera>("shadowmap_camera");
	shadowmap_camera_ptr->set_aspect_ratio(1.0f);
	shadowmap_camera_ptr->set_field_of_view(glm::radians(90.0f));
	shadowmap_camera_ptr->set_near_plane(0.1f);
	shadowmap_camera_ptr->set_far_plane(100.0f);
	shadowmap_camera_ptr->set_node(*light.get_node());
	shadowmap_camera = shadowmap_camera_ptr.get();
	light.get_node()->set_component(*shadowmap_camera_ptr);
	get_scene().add_component(std::move(shadowmap_camera_ptr));

	shadow_render_pipeline = create_shadow_renderpass();
	main_render_pipeline   = create_main_renderpass();

	// Add a GUI with the stats you want to monitor
	get_stats().request_stats({/*stats you require*/});
	create_gui(*window, &get_stats());

	return true;
}

void PointLightShadow::update(float delta_time)
{
	update_scene(delta_time);

	auto &command_buffer = get_render_context().begin();

	command_buffer.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	////////////////////////////////////////////////////////////////////////
	auto &shadow_render_target = *shadow_render_targets_[get_render_context().get_active_frame_index()];
	auto &shadowmap_extent     = shadow_render_target.get_extent();

	set_viewport_and_scissor(command_buffer, shadowmap_extent);
	for (uint32_t i = 0; i < 6; ++i)
	{
		assert(i < shadow_render_targets_[get_render_context().get_active_frame_index()]->get_views().size());
		auto                   &shadowmap = shadow_render_targets_[get_render_context().get_active_frame_index()]->get_views()[i];
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		command_buffer.image_memory_barrier(shadowmap, memory_barrier);
	}
	shadow_render_pipeline->draw(command_buffer, shadow_render_target);
	command_buffer.end_render_pass();

	////////////////////////////////////////////////////////////////////////
	auto &render_target = get_render_context().get_active_frame().get_render_target();
	auto &extent        = render_target.get_extent();
	set_viewport_and_scissor(command_buffer, extent);

	auto &views = get_render_context().get_active_frame().get_render_target().get_views();

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		assert(swapchain_attachment_index < views.size());
		command_buffer.image_memory_barrier(views[swapchain_attachment_index], memory_barrier);
	}

	{
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_UNDEFINED;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		memory_barrier.src_access_mask = 0;
		memory_barrier.dst_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

		assert(depth_attachment_index < views.size());
		command_buffer.image_memory_barrier(views[depth_attachment_index], memory_barrier);
	}

	for (uint32_t i = 0; i < 6; ++i)
	{
		assert(i < shadow_render_targets_[get_render_context().get_active_frame_index()]->get_views().size());
		auto                   &shadowmap = shadow_render_targets_[get_render_context().get_active_frame_index()]->get_views()[i];
		vkb::ImageMemoryBarrier memory_barrier{};
		memory_barrier.old_layout      = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		memory_barrier.new_layout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		memory_barrier.src_access_mask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		memory_barrier.dst_access_mask = VK_ACCESS_SHADER_READ_BIT;
		memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

		command_buffer.image_memory_barrier(shadowmap, memory_barrier);
	}

	main_render_pipeline->draw(command_buffer, render_target);


	command_buffer.end_render_pass();

	vkb::ImageMemoryBarrier memory_barrier{};
	memory_barrier.old_layout      = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	memory_barrier.new_layout      = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	memory_barrier.src_access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	memory_barrier.src_stage_mask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	memory_barrier.dst_stage_mask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

	assert(swapchain_attachment_index < views.size());
	command_buffer.image_memory_barrier(views[swapchain_attachment_index], memory_barrier);
	command_buffer.end();

	get_render_context().submit({command_buffer});
}

void PointLightShadow::create_shadow_render_target(uint32_t size)
{
	shadow_render_targets_.resize(get_render_context().get_render_frames().size());
	shadow_cube_views_.resize(get_render_context().get_render_frames().size());
	depth_cubemap_image_.resize(get_render_context().get_render_frames().size());

	for (uint32_t i = 0; i < shadow_render_targets_.size(); i++)
	{
		VkExtent3D extent{size, size, 1};

		vkb::core::ImageBuilder image_builder{extent};
		VkFormat                depth_format        = vkb::get_suitable_depth_format(get_device().get_gpu().get_handle());
		depth_cubemap_image_[i]               = image_builder.with_format(depth_format)
		                                           .with_usage(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)
		                                           .with_flags(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
		                                           .with_tiling(VK_IMAGE_TILING_OPTIMAL)
		                                           .with_vma_usage(VMA_MEMORY_USAGE_GPU_ONLY)
		                                           .with_array_layers(6)
		                                           .build_unique(get_device());

		shadow_cube_views_[i] = std::make_unique<vkb::core::ImageView>(*depth_cubemap_image_[i], VK_IMAGE_VIEW_TYPE_CUBE);

		std::vector<vkb::core::ImageView> face_image_views;
		for (uint32_t j = 0; j < 6; ++j)
		{
			face_image_views.emplace_back(*depth_cubemap_image_[i], VK_IMAGE_VIEW_TYPE_2D, depth_format, 0, j, 1, 1);
		}
		shadow_render_targets_[i] = std::make_unique<vkb::RenderTarget>(std::move(face_image_views));
	}
}

std::unique_ptr<vkb::RenderPipeline> PointLightShadow::create_main_renderpass()
{
	// Main subpass
	auto main_vs       = vkb::ShaderSource{"shadows/point_main.vert"};
	auto main_fs       = vkb::ShaderSource{"shadows/point_main.frag"};
	auto scene_subpass = std::make_unique<PointMainSubpass>(
	    get_render_context(), std::move(main_vs), std::move(main_fs), get_scene(), *camera, *dynamic_cast<vkb::sg::PerspectiveCamera *>(shadowmap_camera), shadow_cube_views_);

	// Main pipeline
	auto main_render_pipeline = std::make_unique<vkb::RenderPipeline>();
	main_render_pipeline->add_subpass(std::move(scene_subpass));

	return main_render_pipeline;
}

std::unique_ptr<vkb::RenderPipeline> PointLightShadow::create_shadow_renderpass()
{
	// Shadowmap subpass
	auto shadowmap_vs  = vkb::ShaderSource{"shadows/shadowmap.vert"};
	auto shadowmap_fs  = vkb::ShaderSource{"shadows/shadowmap.frag"};
	auto scene_subpass = std::make_unique<PointShadowSubpass>(get_render_context(), std::move(shadowmap_vs), std::move(shadowmap_fs), get_scene(), *shadowmap_camera);

	shadow_subpass = scene_subpass.get();

	// Shadowmap pipeline
	auto shadowmap_render_pipeline = std::make_unique<vkb::RenderPipeline>();
	shadowmap_render_pipeline->add_subpass(std::move(scene_subpass));

	return shadowmap_render_pipeline;
}

std::unique_ptr<vkb::VulkanSampleC> create_point_light_shadow()
{
	return std::make_unique<PointLightShadow>();
}
