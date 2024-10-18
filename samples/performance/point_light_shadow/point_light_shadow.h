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

#pragma once

#include "rendering/render_pipeline.h"
#include "scene_graph/components/perspective_camera.h"
#include "vulkan_sample.h"

constexpr uint32_t SHADOWMAP_RESOLUTION{1024};

struct alignas(16) PointShadowUniform
{
	glm::vec3 light_position;
	float     shadow_far_plane;
};

class PointShadowSubpass : public vkb::GeometrySubpass
{
  public:
	PointShadowSubpass(vkb::RenderContext &render_context,
	                   vkb::ShaderSource &&vertex_source,
	                   vkb::ShaderSource &&fragment_source,
	                   vkb::sg::Scene     &scene,
	                   vkb::sg::Camera    &camera);

	void draw(vkb::CommandBuffer &command_buffer) override;

  protected:
	void update_uniform(vkb::CommandBuffer &command_buffer, vkb::sg::Node &node, size_t thread_index) override;

	void prepare_pipeline_state(vkb::CommandBuffer &command_buffer, VkFrontFace front_face, bool double_sided_material) override;

	vkb::PipelineLayout &prepare_pipeline_layout(vkb::CommandBuffer &command_buffer, const std::vector<vkb::ShaderModule *> &shader_modules) override;

	void prepare_push_constants(vkb::CommandBuffer &command_buffer, vkb::sg::SubMesh &sub_mesh) override;

  private:
	uint32_t                 current_face{0};
	std::array<glm::mat4, 6> view_matrices;
};

class PointMainSubpass : public vkb::ForwardSubpass
{
  public:
	PointMainSubpass(vkb::RenderContext                              &render_context,
	                 vkb::ShaderSource                              &&vertex_source,
	                 vkb::ShaderSource                              &&fragment_source,
	                 vkb::sg::Scene                                  &scene,
	                 vkb::sg::Camera                                 &camera,
	                 vkb::sg::PerspectiveCamera                         &shadowmap_camera,
	                 std::vector<std::unique_ptr<vkb::core::ImageView>> &shadow_cube_views);

	void prepare() override;

	void draw(vkb::CommandBuffer &command_buffer) override;

  private:
	std::unique_ptr<vkb::core::Sampler> cubemap_shadowmap_sampler_{};

	vkb::sg::Camera &shadowmap_camera_;

	std::vector<std::unique_ptr<vkb::core::ImageView>> &shadow_cube_views_;

	PointShadowUniform shadow_uniform_{};
};

class PointLightShadow : public vkb::VulkanSampleC
{
  public:
	PointLightShadow();

	bool prepare(const vkb::ApplicationOptions &options) override;

	void update(float delta_time) override;

	virtual ~PointLightShadow() = default;

  private:
	void create_shadow_render_target(uint32_t size);

	std::unique_ptr<vkb::RenderPipeline> create_main_renderpass();

	std::unique_ptr<vkb::RenderPipeline> create_shadow_renderpass();

	vkb::sg::Camera *camera{};

	PointShadowSubpass *shadow_subpass;

	std::vector < std::unique_ptr<vkb::core::Image>> depth_cubemap_image_;

	std::vector<std::unique_ptr<vkb::RenderTarget>>    shadow_render_targets_;
	std::vector<std::unique_ptr<vkb::core::ImageView>> shadow_cube_views_;
	vkb::sg::Camera                                   *shadowmap_camera{};

	std::unique_ptr<vkb::RenderPipeline> shadow_render_pipeline{};
	std::unique_ptr<vkb::RenderPipeline> main_render_pipeline{};


	uint32_t swapchain_attachment_index{0};

	uint32_t depth_attachment_index{1};
};

std::unique_ptr<vkb::VulkanSampleC> create_point_light_shadow();
