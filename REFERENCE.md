# DAIRlib `sampling_c3` Reference Code

Source: `/root/reference_repos/dairlib_sampling_c3/` (DAIRLab/dairlib). The checkout is on the AUTHORITATIVE branch `push_anything_dev`, HEAD `257e3ede` (see docs/alignment-references.md). NOTE: quotes below were originally extracted at `sampling_based_c3_public` HEAD `b52c68d` — line numbers may have shifted on push_anything_dev; verify against the working tree before relying on exact lines.

This document is a curated quote-and-cite extraction of the reference implementation. Each section gives file:line citations and the actual source so it can be referenced without re-grepping. All commentary is from the dairlib source — none added here.

---

## Tree

```
examples/sampling_c3/
├── franka_sampling_c3_controller.cc    540  C3 MPC process — plans + publishes EE position+force traj
├── franka_osc_controller.cc            277  OSC process — subscribes to traj, tracks via QP
├── franka_joint_osc_controller.cc           Hardware variant (joint-level OSC)
├── franka_sim.cc                            Sim shell
├── generate_samples.{cc,h}             475  Sampling strategies (Perimeter, Shell, …)
├── sampling_c3_utils.{cc,h}             89  AddFrankaToPlant / AddObjectToPlant / AddLCSModelsToPlant
├── reposition.{cc,h}                        Reposition trajectory generator
├── goal_generator.{cc,h}
├── joint_trajectory_generator.{cc,h}
├── parameter_headers/                       YAML loaders
├── urdf/
│   └── end_effector_simple_model.urdf   80  3-DoF floating-EE LCS proxy
├── jacktoy/    push_t/   shared_parameters/

systems/controllers/
├── sampling_based_c3_controller.cc    2171  The MPC + dispatcher + sample buffer
├── sampling_based_c3_controller.h
└── osc/
    ├── operational_space_control.{cc,h}  864  OSC core
    ├── inverse_dynamics_qp.{cc,h}        340  Backend QP: dv/u/λh/λc/λe variables
    ├── external_force_tracking_data.{cc,h} 102 λ_des = traj.value(t) stash
    ├── trans_space_tracking_data.{cc,h}      EE-position tracking objective
    ├── rot_space_tracking_data.{cc,h}        EE-orientation tracking objective
    ├── joint_space_tracking_data.{cc,h}
    ├── osc_tracking_data.{cc,h}
    ├── osc_gains.h
    ├── com_tracking_data.{cc,h}
    ├── end_effector_position.{cc,h}    EE trajectory generator
    ├── end_effector_force.{cc,h}       EE force-trajectory generator
    └── end_effector_orientation.{cc,h}
```

---

## 1 — The 3-DoF floating-EE LCS proxy

### `examples/sampling_c3/urdf/end_effector_simple_model.urdf` (lines 1-80)

```xml
<?xml version="1.0" ?>
<!-- Frames all align with the Franka's origin at configuration = 0. -->

<robot name="end_effector_simple">
  <material name="finger_0_material">
    <color rgba="0.6 0.0 0.0 1.0"/>
  </material>

  <link name="base_link" />

  <link name="end_effector_simple">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.057"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.0195"/>
      </geometry>
      <material name="finger_0_material"/>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.0195"/>
      </geometry>
      <drake:proximity_properties>
        <drake:mu_static value="1"/>
        <drake:mu_dynamic value="1"/>
      </drake:proximity_properties>
    </collision>
  </link>

  <link name="end_effector_simple_fake_x" />

  <link name="end_effector_simple_fake_y" />

  <joint name="end_effector_simple_to_base_x" type="prismatic">
    <parent link="base_link"/>
    <child link="end_effector_simple_fake_x"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="end_effector_simple_to_base_y" type="prismatic">
    <parent link="end_effector_simple_fake_x"/>
    <child link="end_effector_simple_fake_y"/>
    <axis xyz="0 1 0"/>
  </joint>
    <joint name="end_effector_simple_to_base_z" type="prismatic">
    <parent link="end_effector_simple_fake_y"/>
    <child link="end_effector_simple"/>
    <axis xyz="0 0 1"/>
  </joint>

   <transmission name="end_effector_simple_to_base_trans_x"
    type="SimpleTransmission">
    <actuator name="end_effector_simple_to_base_act_x"/>
    <joint name="end_effector_simple_to_base_x"/>
    <mechanicalReduction>1</mechanicalReduction>
  </transmission>

     <transmission name="end_effector_simple_to_base_trans_y"
      type="SimpleTransmission">
    <actuator name="end_effector_simple_to_base_act_y"/>
    <joint name="end_effector_simple_to_base_y"/>
    <mechanicalReduction>1</mechanicalReduction>
  </transmission>

     <transmission name="end_effector_simple_to_base_trans_z"
      type="SimpleTransmission">
    <actuator name="end_effector_simple_to_base_act_z"/>
    <joint name="end_effector_simple_to_base_z"/>
    <mechanicalReduction>1</mechanicalReduction>
  </transmission>

  <drake:collision_filter_group name="finger_0_group">
  <drake:member link="end_effector_simple"/>
  <drake:ignored_collision_filter_group name="finger_0_group"/>
  </drake:collision_filter_group>
</robot>
```

EE: 57 g sphere of radius 19.5 mm, μ=1, on three prismatic joints x/y/z, each with a `<transmission>` block ⇒ **3 actuators**. The LCS plant's `n_u = 3` and `u_sol` is a 3-D Cartesian force command.

### `examples/sampling_c3/sampling_c3_utils.cc:64-87` — `AddLCSModelsToPlant`

```cpp
void AddLCSModelsToPlant(
    MultibodyPlant<double>* plant,
    SceneGraph<double>* scene_graph,
    const std::string& object_model,
    const bool& include_end_effector_orientation) {
  // Cannot currently handle end effector orientation (would just require new
  // EE simple model with orientation DOFs).
  DRAKE_DEMAND(!include_end_effector_orientation);

  Parser parser_lcs(plant);
  parser_lcs.SetAutoRenaming(true);
  parser_lcs.AddModels(kEndEffectorSimpleModel);
  parser_lcs.AddModels(kGroundModel);
  parser_lcs.AddModels(object_model);

  RigidTransform<double> X_WI = RigidTransform<double>::Identity();

  RigidTransform<double> X_W_G = RigidTransform<double>(
      drake::math::RotationMatrix<double>(), kWorldToGroundOffset);
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("base_link"), X_WI);
  plant->WeldFrames(plant->world_frame(),
                    plant->GetFrameByName("ground"), X_W_G);
}
```

### `examples/sampling_c3/franka_sampling_c3_controller.cc:87-100` — LCS plant build

```cpp
// Create the LCS plant containing a floating EE, object, and ground.
DiagramBuilder<double> plant_lcs_builder;
auto [plant_lcs, scene_graph] =
  AddMultibodyPlantSceneGraph(&plant_lcs_builder, 0.0);
AddLCSModelsToPlant(&plant_lcs, &scene_graph, controller_params.object_model,
                    controller_params.include_end_effector_orientation);
plant_lcs.Finalize();
```

---

## 2 — Force-trajectory publish (C3 controller → OSC)

### `systems/controllers/sampling_based_c3_controller.cc:1493-1520` — `u_sol_` → `"end_effector_force_target"`

```cpp
MatrixXd knots = MatrixXd::Zero(6, N_);
knots.topRows(3) = c3_solution->x_sol_.topRows(3).cast<double>();
knots.bottomRows(3) =
  c3_solution->x_sol_.bottomRows(n_v_).topRows(3).cast<double>();

LcmTrajectory::Trajectory end_effector_traj;
end_effector_traj.traj_name = "end_effector_position_target";
end_effector_traj.datatypes =
  std::vector<std::string>(knots.rows(), "double");
end_effector_traj.datapoints = knots;
end_effector_traj.time_vector = c3_solution->time_vector_.cast<double>();
LcmTrajectory lcm_traj({end_effector_traj}, {"end_effector_position_target"},
                       "end_effector_position_target",
                       "end_effector_position_target", false);

MatrixXd force_samples = c3_solution->u_sol_.cast<double>();
LcmTrajectory::Trajectory force_traj;
force_traj.traj_name = "end_effector_force_target";
force_traj.datatypes =
  std::vector<std::string>(force_samples.rows(), "double");
force_traj.datapoints = force_samples;
force_traj.time_vector = c3_solution->time_vector_.cast<double>();
lcm_traj.AddTrajectory(force_traj.traj_name, force_traj);

output->saved_traj = lcm_traj.GenerateLcmObject();
```

`u_sol_` is the C3 solver's commanded control input on the floating-EE LCS — a 3 × N matrix of Cartesian forces. It is published as the `"end_effector_force_target"` trajectory over LCM channel `tracking_trajectory_actor_channel`.

---

## 3 — OSC controller — tracking-data registration

### `examples/sampling_c3/franka_osc_controller.cc:65-110` — process setup

```cpp
int DoMain(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::lcm::DrakeLcm lcm(FLAGS_lcm_url);
  …
  // Create a Franka-only plant.
  drake::multibody::MultibodyPlant<double> plant(0.0);
  AddFrankaToPlant(&plant);
  plant.Finalize();
  auto plant_context = plant.CreateDefaultContext();

  // Piece together the diagram.
  DiagramBuilder<double> builder;

  auto state_receiver = builder.AddSystem<systems::RobotOutputReceiver>(plant);
  auto end_effector_trajectory_sub = builder.AddSystem(
      LcmSubscriberSystem::Make<dairlib::lcmt_timestamped_saved_traj>(
          lcm_channel_params.tracking_trajectory_actor_channel, &lcm));
  auto end_effector_position_receiver =
      builder.AddSystem<systems::LcmTrajectoryReceiver>(
          "end_effector_position_target");
  auto end_effector_force_receiver =
      builder.AddSystem<systems::LcmTrajectoryReceiver>(
          "end_effector_force_target");
  auto end_effector_orientation_receiver =
      builder.AddSystem<systems::LcmOrientationTrajectoryReceiver>(
          "end_effector_orientation_target");
```

Note: OSC's `MultibodyPlant` is **Franka-only** (no box), distinct from the C3 controller's LCS plant (floating EE + object + ground).

### `examples/sampling_c3/franka_osc_controller.cc:138-200` — tracking-data registration

```cpp
  auto osc = builder.AddSystem<systems::controllers::OperationalSpaceControl>(
      plant, plant_context.get(), false);
  …

  auto end_effector_position_tracking_data =
      std::make_unique<TransTaskSpaceTrackingData>(
          "end_effector_target", osc_params.K_p_end_effector,
          osc_params.K_d_end_effector, osc_params.W_end_effector,
          plant, plant);
  end_effector_position_tracking_data->AddPointToTrack(kEndEffectorName);
  const VectorXd& end_effector_acceleration_limits =
      osc_params.end_effector_acceleration * Vector3d::Ones();
  end_effector_position_tracking_data->SetCmdAccelerationBounds(
      -end_effector_acceleration_limits, end_effector_acceleration_limits);
  auto mid_link_position_tracking_data_for_rel =
      std::make_unique<JointSpaceTrackingData>(
          "panda_joint2_target", osc_params.K_p_mid_link,
          osc_params.K_d_mid_link, osc_params.W_mid_link, plant,
          plant);
  mid_link_position_tracking_data_for_rel->AddJointToTrack("panda_joint2",
                                                           "panda_joint2dot");

  auto end_effector_force_tracking_data =
      std::make_unique<ExternalForceTrackingData>(
          "end_effector_force", osc_params.W_ee_lambda, plant, plant,
          kEndEffectorName, Vector3d::Zero());

  auto end_effector_orientation_tracking_data =
      std::make_unique<RotTaskSpaceTrackingData>(
          "end_effector_orientation_target",
          osc_params.K_p_end_effector_rot,
          osc_params.K_d_end_effector_rot,
          osc_params.W_end_effector_rot, plant, plant);
  end_effector_orientation_tracking_data->AddFrameToTrack(kEndEffectorName);
  Eigen::VectorXd orientation_target = Eigen::VectorXd::Zero(4);
  orientation_target(0) = 1;
  osc->AddTrackingData(std::move(end_effector_position_tracking_data));
  // Since the Franka has 7 joints to control a 6 DOF EE command, add an
  // additional tracking objective for joint 2 at a good configuration for the
  // sampling C3 experiments.  1.1 joint target empirically works well.
  osc->AddConstTrackingData(std::move(mid_link_position_tracking_data_for_rel),
                            1.1 * VectorXd::Ones(1));
  osc->AddTrackingData(std::move(end_effector_orientation_tracking_data));
  osc->AddForceTrackingData(std::move(end_effector_force_tracking_data));
  osc->SetAccelerationCostWeights(osc_params.W_acceleration);
  osc->SetInputCostWeights(osc_params.W_input_regularization);
  osc->SetInputSmoothingCostWeights(osc_params.W_input_smoothing_regularization);
  if (osc_params.enforce_acceleration_constraints) {
    osc->EnableAccelerationConstraints();
  } else {
    osc->DisableAccelerationConstraints();
  }
  osc->SetContactFriction(osc_params.mu);
  osc->SetOsqpSolverOptions(solver_options);

  osc->Build();
```

Three simultaneous tracking objectives:
1. `TransTaskSpaceTrackingData("end_effector_target", K_p, K_d, W_end_effector)` — EE position (3-D)
2. `RotTaskSpaceTrackingData("end_effector_orientation_target", …)` — EE orientation
3. `ExternalForceTrackingData("end_effector_force", W_ee_lambda, …)` — EE Cartesian force

Plus a `JointSpaceTrackingData` constant pin on `panda_joint2 = 1.1 rad`.

### `examples/sampling_c3/franka_osc_controller.cc:237-261` — diagram wiring (subscribe trajectories → tracking ports)

```cpp
  builder.Connect(state_receiver->get_output_port(0),
                  osc->get_input_port_robot_output());
  builder.Connect(end_effector_trajectory_sub->get_output_port(),
                  end_effector_position_receiver->get_input_port_trajectory());
  builder.Connect(end_effector_trajectory_sub->get_output_port(),
                  end_effector_force_receiver->get_input_port_trajectory());
  builder.Connect(
      end_effector_trajectory_sub->get_output_port(),
      end_effector_orientation_receiver->get_input_port_trajectory());
  builder.Connect(end_effector_position_receiver->get_output_port(0),
                  end_effector_trajectory->get_input_port_trajectory());
  builder.Connect(state_receiver->get_output_port(0),
                  end_effector_trajectory->get_input_port_state());
  builder.Connect(
      end_effector_orientation_receiver->get_output_port(0),
      end_effector_orientation_trajectory->get_input_port_trajectory());
  builder.Connect(end_effector_trajectory->get_output_port(0),
                  osc->get_input_port_tracking_data("end_effector_target"));
  builder.Connect(
      end_effector_orientation_trajectory->get_output_port(0),
      osc->get_input_port_tracking_data("end_effector_orientation_target"));
  builder.Connect(end_effector_force_receiver->get_output_port(0),
                  end_effector_force_trajectory->get_input_port_trajectory());
  builder.Connect(end_effector_force_trajectory->get_output_port(0),
                  osc->get_input_port_tracking_data("end_effector_force"));
```

---

## 4 — `ExternalForceTrackingData` — the force-tracking objective

### `systems/controllers/osc/external_force_tracking_data.h` (full)

```cpp
#pragma once

#include <drake/common/trajectories/trajectory.h>
#include <drake/multibody/plant/multibody_plant.h>

namespace dairlib {
namespace systems {
namespace controllers {

/// ExternalForceTrackingData
/// Force tracking objective. Used to track desired external forces. Requires
/// contact points on the MultibodyPlant where contact forces enter the dynamics
class ExternalForceTrackingData {
 public:
  ExternalForceTrackingData(
      const std::string& name, const Eigen::MatrixXd& W,
      const drake::multibody::MultibodyPlant<double>& plant_w_spr,
      const drake::multibody::MultibodyPlant<double>& plant_wo_spr,
      const std::string& body_name, const Eigen::Vector3d& pt_on_body);

  const Eigen::MatrixXd& GetWeight() const { return W_; }
  const Eigen::VectorXd& GetLambdaDes() const { return lambda_des_; }
  const std::string& GetName() const { return name_; };
  const Eigen::Vector3d& GetPointOnBody() const { return pt_on_body_; };
  const drake::multibody::Frame<double>& GetBodyFrame() const {
    return *body_frame_wo_spr_;
  };

  const drake::multibody::MultibodyPlant<double>& plant_w_spr() const {
    return plant_w_spr_;
  };
  const drake::multibody::MultibodyPlant<double>& plant_wo_spr() const {
    return plant_wo_spr_;
  };
  void Update(const Eigen::VectorXd& x_w_spr,
              const drake::systems::Context<double>& context_w_spr,
              const Eigen::VectorXd& x_wo_spr,
              const drake::systems::Context<double>& context_wo_spr,
              const drake::trajectories::Trajectory<double>& traj, double t);

 protected:
 private:
  std::string name_;

  const drake::multibody::MultibodyPlant<double>& plant_w_spr_;
  const drake::multibody::MultibodyPlant<double>& plant_wo_spr_;
  // World frames
  const drake::multibody::RigidBodyFrame<double>& world_w_spr_;
  const drake::multibody::RigidBodyFrame<double>& world_wo_spr_;

  const drake::multibody::RigidBodyFrame<double>* body_frame_w_spr_;
  const drake::multibody::RigidBodyFrame<double>* body_frame_wo_spr_;
  const Eigen::Vector3d pt_on_body_;

  Eigen::VectorXd lambda_des_;
  Eigen::MatrixXd W_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace dairlib
```

### `systems/controllers/osc/external_force_tracking_data.cc` (full)

```cpp
#include "external_force_tracking_data.h"

using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::string;
using std::vector;

using drake::multibody::JacobianWrtVariable;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;

namespace dairlib::systems::controllers {

ExternalForceTrackingData::ExternalForceTrackingData(
    const string& name, const MatrixXd& W,
    const MultibodyPlant<double>& plant_w_spr,
    const MultibodyPlant<double>& plant_wo_spr, const std::string& body_name,
    const Vector3d& pt_on_body)
    : name_(name),
      plant_w_spr_(plant_w_spr),
      plant_wo_spr_(plant_wo_spr),
      world_w_spr_(plant_w_spr_.world_frame()),
      world_wo_spr_(plant_wo_spr_.world_frame()),
      body_frame_w_spr_(&plant_w_spr_.GetBodyByName(body_name).body_frame()),
      body_frame_wo_spr_(&plant_wo_spr_.GetBodyByName(body_name).body_frame()),
      pt_on_body_(pt_on_body),
      W_(W) {
  lambda_des_ = Vector3d::Zero();
}

void ExternalForceTrackingData::Update(
    const Eigen::VectorXd& x_w_spr,
    const drake::systems::Context<double>& context_w_spr,
    const Eigen::VectorXd& x_wo_spr,
    const drake::systems::Context<double>& context_wo_spr,
    const drake::trajectories::Trajectory<double>& traj, double t) {
  DRAKE_DEMAND(traj.rows() == 3);
  lambda_des_ = traj.value(t);
}

}  // namespace dairlib::systems::controllers
```

`lambda_des_` is a 3-D Cartesian force, refreshed every OSC tick from `traj.value(t)` where `traj` is the `end_effector_force_target` trajectory.

---

## 5 — `InverseDynamicsQp` — backend QP

### `systems/controllers/osc/inverse_dynamics_qp.h:23-50` — the QP form

```cpp
/*!
 * Wrapper class for handling kinematics and dynamics for a quadratic program
 * which computes dynamically consistent accelerations, inputs,
 * and constraint forces (including contacts) which minimize some combined
 * cost on these variables.
 *
 * Designed to be used as a back-end for operational space control, but
 * applicable to other model-based instantaneous QP controllers.
 *
 * Constructs a QP of the form
 *
 * minimize C₁(v̇) + C₂(u), C₃(λₕ) + C₄(λ_c) + C₄(λₑ)
 * subject to Jₕv̇ + J̇ₕ = 0
 *            J_cv̇ + J̇_c = 0
 *            Mv̇ + c = Bu + Jₕᵀλₕ + J_cᵀλ_c + Jₑᵀλₑ
 *            λ_c ∈ FrictionCone
 *
 * Where
 *
 * Cᵢ are arbitrary user-defined convex quadratic costs,
 * v̇ are generalized accelerations,
 * u are actuation efforts,
 * λₕ are constraint forces for general holonomic constraints such linkages,
 * λ_c are constraint forces for contact constraints
 * λₑ are external forces, such as contact forces, which do not constrain the
 * robot's motion (like force on the end effector of a robot arm)
 *
 */
class InverseDynamicsQp {
```

### `systems/controllers/osc/inverse_dynamics_qp.h:63-87` — `AddExternalForce` signature

```cpp
  /*!
   * Adds an external force to the dynamics.
   *
   * Note: By default, no constraints are imposed on the external forces. If
   * external forces are declared without corresponding constraints/costs
   * from the upstream controller, then unexpected behavior may occur.
   * (i.e. Unconstrained forces effectively act as input variables.)
   *
   * @param name name of the external force
   * @param eval WorldPointEvaluator for the associated jacobian
   */
  void AddExternalForce(
      const std::string& name,
      std::unique_ptr<const multibody::WorldPointEvaluator<double>> eval);
```

### `systems/controllers/osc/inverse_dynamics_qp.cc:63-73` — `AddExternalForce` impl

```cpp
void InverseDynamicsQp::AddExternalForce(
    const string& name, unique_ptr<const WorldPointEvaluator<double>> eval) {
  DRAKE_DEMAND(not built_);
  DRAKE_DEMAND(&eval->plant() == &plant_);
  DRAKE_DEMAND(external_force_evaluators_.count(name) == 0);

  external_force_evaluators_.insert({name, std::move(eval)});
  lambda_e_start_and_size_.insert(
      {name, {ne_, external_force_evaluators_.at(name)->num_full()}});
  ne_ += external_force_evaluators_.at(name)->num_full();
}
```

### `systems/controllers/osc/inverse_dynamics_qp.cc:75-146` — `Build` (decision variables + dynamics shell + friction cones + bounds)

```cpp
void InverseDynamicsQp::Build() {
  DRAKE_DEMAND(not built_);

  dv_ = prog_.NewContinuousVariables(nv_, "dv");
  u_ = prog_.NewContinuousVariables(nu_, "u");
  lambda_h_ = prog_.NewContinuousVariables(nh_, "lambda_holonomic");
  lambda_c_ = prog_.NewContinuousVariables(nc_, "lambda_contact");
  lambda_e_ = prog_.NewContinuousVariables(ne_, "lambda_external");
  epsilon_ = prog_.NewContinuousVariables(nc_active_, "soft_constraint_slack");

  dynamics_c_ =
      prog_
          .AddLinearEqualityConstraint(
              MatrixXd::Zero(nv_, nv_ + nu_ + nh_ + nc_ + ne_),
              VectorXd::Zero(nv_), {dv_, u_, lambda_h_, lambda_c_, lambda_e_})
          .evaluator();

  holonomic_c_ = prog_
                     .AddLinearEqualityConstraint(MatrixXd::Zero(nh_, nv_),
                                                  VectorXd::Zero(nh_), dv_)
                     .evaluator();

  contact_c_ = prog_
                   .AddLinearEqualityConstraint(
                       MatrixXd::Zero(nc_active_, nv_ + nc_active_),
                       VectorXd::Zero(nc_active_), {dv_, epsilon_})
                   .evaluator();

  for (const auto& [cname, eval] : contact_constraint_evaluators_) {
    double mu = mu_map_.at(cname);
    MatrixXd A = MatrixXd(5, 3);
    A << -1, 0, mu, 0, -1, mu, 1, 0, mu, 0, 1, mu, 0, 0, 1;
    lambda_c_friction_cone_.insert(
        {cname,
         prog_
             .AddLinearConstraint(
                 A, VectorXd::Zero(5),
                 VectorXd::Constant(5, std::numeric_limits<double>::infinity()),
                 lambda_c_.segment(lambda_c_start_.at(cname), 3))
             .evaluator()});
  }

  if (with_input_constraints_) {
    VectorXd u_min(nu_);
    VectorXd u_max(nu_);
    for (drake::multibody::JointActuatorIndex i(0); i < nu_; ++i) {
      u_min[i] = -plant_.get_joint_actuator(i).effort_limit();
      u_max[i] = plant_.get_joint_actuator(i).effort_limit();
    }
    input_limit_c_ =
        prog_.AddBoundingBoxConstraint(u_min, u_max, u_).evaluator();
  }

  if (with_acceleration_constraints_) {
    VectorXd ddq_min = VectorXd::Zero(nq_);
    VectorXd ddq_max = VectorXd::Zero(nq_);
    for (drake::multibody::JointIndex i(0); i < nq_; ++i) {
      if (plant_.get_joint(i).acceleration_lower_limits().size() != 0) {
        ddq_min[i] = plant_.get_joint(i).acceleration_lower_limits()[0];
        ddq_max[i] = plant_.get_joint(i).acceleration_upper_limits()[0];
      }
    }
    if (ddq_max.isZero()) {
      throw std::runtime_error(
          "Attempting to set acceleration limits when acceleration limits have "
          "not been defined for the plant.");
    }
    acceleration_limit_c_ =
        prog_.AddBoundingBoxConstraint(ddq_min, ddq_max, dv_).evaluator();
  }
  built_ = true;
}
```

Decision variables: `dv_` (n_v), `u_` (n_u), `lambda_h_` (n_h holonomic), `lambda_c_` (n_c contact), `lambda_e_` (n_e external), `epsilon_` (n_c_active soft slack). Friction cone is the 5-row stack `[−x+μz, −y+μz, x+μz, y+μz, z] ≥ 0` per contact.

### `systems/controllers/osc/inverse_dynamics_qp.cc:164-228` — `UpdateDynamics` (assemble `M v̇ − B u − Jhᵀλh − Jcᵀλc − Jeᵀλe = −bias`)

```cpp
void InverseDynamicsQp::UpdateDynamics(
    const VectorXd& x, const vector<string>& active_contact_constraints,
    const vector<string>& active_external_forces) {
  SetPositionsAndVelocitiesIfNew<double>(plant_, x, context_);

  MatrixXd M(nv_, nv_);
  VectorXd bias(nv_);
  MatrixXd B = plant_.MakeActuationMatrix();
  VectorXd grav = plant_.CalcGravityGeneralizedForces(*context_);

  plant_.CalcMassMatrix(*context_, &M);
  plant_.CalcBiasTerm(*context_, &bias);

  if (with_gravity_compensation_) {
    bias = bias - grav;
  }

  MatrixXd Jh = MatrixXd::Zero(nh_, nv_);
  VectorXd Jh_dot_v = VectorXd::Zero(nh_);

  if (holonomic_constraints_ != nullptr) {
    Jh = holonomic_constraints_->EvalFullJacobian(*context_);
    Jh_dot_v = holonomic_constraints_->EvalFullJacobianDotTimesV(*context_);
  }

  MatrixXd Jc_active = MatrixXd::Zero(nc_active_, nv_);
  VectorXd Jc_active_dot_v = VectorXd::Zero(nc_active_);
  MatrixXd Jc = MatrixXd::Zero(nc_, nv_);
  MatrixXd Je = MatrixXd::Zero(ne_, nv_);

  for (const auto& c : active_contact_constraints) {
    DRAKE_DEMAND(contact_constraint_evaluators_.count(c) > 0);
    const auto& evaluator = contact_constraint_evaluators_.at(c);
    Jc.block(lambda_c_start_.at(c), 0, 3, nv_) =
        evaluator->EvalFullJacobian(*context_);
    int start = Jc_active_start_.at(c);
    for (int i = 0; i < evaluator->num_active(); ++i) {
      Jc_active.row(start + i) =
          Jc.row(lambda_c_start_.at(c) + evaluator->active_inds().at(i));
      Jc_active_dot_v.segment(start, evaluator->num_active()) =
          evaluator->EvalActiveJacobianDotTimesV(*context_);
    }
  }
  for (const auto& e : active_external_forces) {
    const auto& [start, size] = lambda_e_start_and_size_.at(e);
    Je.block(start, 0, size, nv_) =
        external_force_evaluators_.at(e)->EvalFullJacobian(*context_);
  }

  MatrixXd A_dyn = MatrixXd::Zero(nv_, nv_ + nu_ + nh_ + nc_ + ne_);
  A_dyn.block(0, 0, nv_, nv_) = M;
  A_dyn.block(0, nv_, nv_, nu_) = -B;
  A_dyn.block(0, nv_ + nu_, nv_, nh_) = -Jh.transpose();
  A_dyn.block(0, nv_ + nu_ + nh_, nv_, nc_) = -Jc.transpose();
  A_dyn.block(0, nv_ + nu_ + nh_ + nc_, nv_, ne_) = -Je.transpose();

  MatrixXd A_c = MatrixXd::Zero(nc_active_, nv_ + nc_active_);
  A_c.block(0, 0, nc_active_, nv_) = Jc_active;
  A_c.block(0, nv_, nc_active_, nc_active_) =
      MatrixXd::Identity(nc_active_, nc_active_);

  dynamics_c_->UpdateCoefficients(A_dyn, -bias);
  holonomic_c_->UpdateCoefficients(Jh, -Jh_dot_v);
  contact_c_->UpdateCoefficients(A_c, -Jc_active_dot_v);
}
```

### `systems/controllers/osc/inverse_dynamics_qp.h:203-206` — `UpdateCost`

```cpp
  void UpdateCost(const std::string& name, const Eigen::MatrixXd& Q,
                  const Eigen::VectorXd& b, double c = 0) {
    all_costs_.at(name)->UpdateCoefficients(Q, b, c, true);
  };
```

---

## 6 — `OperationalSpaceControl` — high-level OSC

### `systems/controllers/osc/operational_space_control.cc:203-222` — `AddForceTrackingData` (wire to `id_qp_`)

```cpp
// Tracking data methods
void OperationalSpaceControl::AddForceTrackingData(
    std::unique_ptr<ExternalForceTrackingData> tracking_data) {
  force_tracking_data_vec_->push_back(std::move(tracking_data));

  // Declare point where external force is applied in the world frame to the OSC
  // backend
  auto evaluator = std::make_unique<WorldPointEvaluator<double>>(
      plant_, force_tracking_data_vec_->back()->GetPointOnBody(),
      force_tracking_data_vec_->back()->GetBodyFrame());
  id_qp_.AddExternalForce(force_tracking_data_vec_->back()->GetName(),
                          std::move(evaluator));

  // Construct input ports and add element to traj_name_to_port_index_map_ if
  // the port for the traj is not created yet
  string traj_name = force_tracking_data_vec_->back()->GetName();
  if (traj_name_to_port_index_map_.find(traj_name) ==
      traj_name_to_port_index_map_.end()) {
    PiecewisePolynomial<double> pp = PiecewisePolynomial<double>();
    int port_index =
        this->DeclareAbstractInputPort(
                traj_name,
                drake::Value<drake::trajectories::Trajectory<double>>(pp))
            .get_index();
```

### `systems/controllers/osc/operational_space_control.cc:330-340` — track-external-force cost slot reservation

```cpp
  // 5. Track external force cost
  int ne = id_qp_.lambda_e().rows();
  for (const auto& data : *force_tracking_data_vec_) {
    id_qp_.AddQuadraticCost(data->GetName(), MatrixXd::Zero(ne, ne),
                            VectorXd::Zero(ne), id_qp_.lambda_e());
  }
```

### `systems/controllers/osc/operational_space_control.cc:401-486` — per-tick `UpdateDynamics` + tracking-cost + force-tracking-cost update

```cpp
  const auto active_contact_names = contact_names_map_.count(fsm_state) > 0
                                        ? contact_names_map_.at(fsm_state)
                                        : std::vector<std::string>();
  std::vector<std::string> active_external_forces = {};
  for (const auto& force_tracking_data : *force_tracking_data_vec_) {
    active_external_forces.push_back(force_tracking_data->GetName());
  }
  id_qp_.UpdateDynamics(x_w_spr, active_contact_names, active_external_forces);

  …

  // Update costs
  // 4. Tracking cost
  for (unsigned int i = 0; i < tracking_data_vec_->size(); i++) {
    auto tracking_data = tracking_data_vec_->at(i).get();

    if (tracking_data->IsActive(fsm_state)) {
      …
      const VectorXd& ddy_t = tracking_data->GetYddotCommand();
      const MatrixXd& W = tracking_data->GetWeight();
      const MatrixXd& J_t = tracking_data->GetJ();
      const VectorXd& JdotV_t = tracking_data->GetJdotTimesV();
      const VectorXd constant_term = (JdotV_t - ddy_t);

      id_qp_.UpdateCost(tracking_data->GetName(), 2 * J_t.transpose() * W * J_t,
                        2 * J_t.transpose() * W * (JdotV_t - ddy_t),
                        constant_term.transpose() * W * constant_term);
    } else {
      id_qp_.UpdateCost(tracking_data->GetName(), MatrixXd::Zero(n_v_, n_v_),
                        VectorXd::Zero(n_v_));
    }
  }

  // Update tracking cost for external forces
  for (auto& force_tracking_data : *force_tracking_data_vec_) {
    int port_index =
        traj_name_to_port_index_map_.at(force_tracking_data->GetName());
    const drake::AbstractValue* input_traj =
        this->EvalAbstractInput(context, port_index);
    DRAKE_DEMAND(input_traj != nullptr);
    const auto& traj =
        input_traj->get_value<drake::trajectories::Trajectory<double>>();
    force_tracking_data->Update(x_w_spr, *context_, x_wo_spr, *context_, traj,
                                t);
    const MatrixXd W = force_tracking_data->GetWeight();
    const VectorXd lambda_des = force_tracking_data->GetLambdaDes();
    id_qp_.UpdateCost(force_tracking_data->GetName(), 2 * W,
                      -2 * W * lambda_des,
                      lambda_des.transpose() * W * lambda_des);
  }
```

Three things happen every OSC tick:
1. **Position task cost**: `‖J_t v̇ + J̇_t v − ÿ_des‖²_W` → `Q = 2 J_t^T W J_t`, `b = 2 J_t^T W (J̇_t v − ÿ_des)`.
2. **Force task cost**: `‖λ_ext − λ_des‖²_W` → `Q = 2W`, `b = −2W λ_des`, on the `lambda_e_` block.
3. **Dynamics update**: `M v̇ + bias = B u + Jhᵀλh + Jcᵀλc + Jeᵀλe`, refreshed for current `q,v` plus the active contact + external-force sets.

The QP then solves jointly for `(v̇, u, λh, λc, λe, ε)` minimizing the summed costs subject to dynamics + friction cone + bounds.

---

## 7 — Sample generation

### `examples/sampling_c3/generate_samples.h:22-39` — public `GenerateSampleStates`

```cpp
/// Public function
std::vector<Eigen::VectorXd> GenerateSampleStates(
    const int& n_q,
    const int& n_v,
    const int& n_u,
    const Eigen::VectorXd& x_lcs,
    const bool& is_doing_c3,
    const SamplingParams& sampling_params,
    const SamplingC3Options& sampling_c3_options,
    drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>* context_ad,
    const std::vector<
        std::vector<drake::SortedPair<drake::geometry::GeometryId>>>&
        contact_geoms
);
```

### `examples/sampling_c3/generate_samples.cc:24-123` — strategy dispatch

```cpp
std::vector<Eigen::VectorXd> GenerateSampleStates(
    const int& n_q, const int& n_v, const int& n_u,
    const Eigen::VectorXd& x_lcs, const bool& is_doing_c3,
    const SamplingParams& sampling_params,
    const SamplingC3Options& sampling_c3_options,
    drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>* context_ad,
    const std::vector<
        std::vector<drake::SortedPair<drake::geometry::GeometryId>>>&
        contact_geoms) {
  // Determine number of samples based on mode.
  int num_samples;
  if (is_doing_c3) {
    num_samples = sampling_params.num_additional_samples_c3;
  } else {
    num_samples = sampling_params.num_additional_samples_repos;
  }
  std::vector<Eigen::VectorXd> candidate_states(num_samples);
  // Initialize all candidate states to be the same as the current LCS state.
  // NOTE:  A naive step might be to set the sample EE velocities to zero, but
  // in practice this can cause undesired cost differences between the current
  // location and the candidate states.  Keeping the current EE velocity the
  // same across all samples is an equalizer.
  for (int i = 0; i < num_samples; i++) {
    candidate_states[i] = x_lcs;
  }

  // Split function calls based on sampling strategy.
  SamplingStrategy strategy = sampling_params.sampling_strategy;
  if (strategy == SamplingStrategy::kRadiallySymmetric) {
    for (int i = 0; i < num_samples; i++) {
      candidate_states[i].head(3) = RadiallySymmetricSampling(
        n_q, n_v, x_lcs, num_samples, i, sampling_params.sampling_radius,
        sampling_params.sampling_height);
      …
    }
  } else if (strategy == SamplingStrategy::kRandomOnCircle) {
    …
  } else if (strategy == SamplingStrategy::kRandomOnSphere) {
    …
  } else if (strategy == SamplingStrategy::kFixed) {
    …
  } else if (strategy == SamplingStrategy::kRandomOnPerimeter) {
    for (int i = 0; i < num_samples; i++) {
      do {
        candidate_states[i].head(3) = PerimeterSampling(
          n_q, n_v, n_u, x_lcs, plant, context, plant_ad, context_ad,
          contact_geoms, sampling_params, sampling_c3_options);
      } while (sampling_params.filter_samples_for_safety &&
               !IsSampleInWorkspace(candidate_states[i], sampling_c3_options));
    }
  } else if (strategy == SamplingStrategy::kRandomOnShell) {
    for (int i = 0; i < num_samples; i++) {
      do {
        candidate_states[i].head(3) = ShellSampling(
          n_q, n_v, n_u, x_lcs, plant, context, plant_ad, context_ad,
          contact_geoms, sampling_params, sampling_c3_options);
      } while (sampling_params.filter_samples_for_safety &&
               !IsSampleInWorkspace(candidate_states[i], sampling_c3_options));
    }
  } else {
    throw std::runtime_error("Error:  Sampling strategy not recognized.");
  }
  return candidate_states;
}
```

Available strategies: `kRadiallySymmetric`, `kRandomOnCircle`, `kRandomOnSphere`, `kFixed`, `kRandomOnPerimeter`, `kRandomOnShell`. `c3` and `repos` modes can request different sample counts (`num_additional_samples_c3` vs `num_additional_samples_repos`).

### `examples/sampling_c3/generate_samples.cc:210-282` — `PerimeterSampling` (body-frame XY box on a fixed z-plane, projected outside obj surface)

```cpp
Eigen::Vector3d PerimeterSampling(
    const int& n_q, const int& n_v, const int& n_u,
    const Eigen::VectorXd& x_lcs,
    drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>* context_ad,
    const std::vector<
        std::vector<drake::SortedPair<drake::geometry::GeometryId>>>&
        contact_geoms,
    const SamplingParams& sampling_params,
    const SamplingC3Options sampling_c3_options)
{
  Eigen::VectorXd candidate_state = VectorXd::Zero(n_q + n_v);
  int min_distance_index = -1;

  // Try projecting colliding samples until one is near desired sampling height
  // and maintains the desired clearance.
  while (true) {
    do {
      // These are in body frame.
      double x_sample = RandomUniform(sampling_params.grid_x_limits[0],
                                      sampling_params.grid_x_limits[1]);
      double y_sample = RandomUniform(sampling_params.grid_y_limits[0],
                                      sampling_params.grid_y_limits[1]);
      // WARNING:  This assumes 1) the body's z-axis is roughly aligned with the
      // world z-axis, and 2) the body origin is roughly at the mid-height of
      // the object.
      double z_sample = 0;

      // Convert to world frame using the current object state.
      Eigen::Quaterniond quat_object(x_lcs(3), x_lcs(4), x_lcs(5), x_lcs(6));
      Eigen::Vector3d object_position = x_lcs.segment(7, 3);
      candidate_state = x_lcs;
      candidate_state.head(3) =
          quat_object * Eigen::Vector3d(x_sample, y_sample, z_sample) +
          object_position;

      // Project samples to specified sampling height in world frame.
      candidate_state[2] = sampling_params.sampling_height;
    } while (!IsSampleWithinDistanceOfSurface(
      n_q, n_v, n_u, 0.0, candidate_state, plant, context, plant_ad, context_ad,
      contact_geoms, sampling_c3_options, min_distance_index));

    // Project the sample past the surface of the object with clearance.
    Eigen::VectorXd projected_state = ProjectSampleOutsideObject(
      candidate_state, min_distance_index, sampling_params, plant, *context,
      contact_geoms);

    // Check the desired clearance is satisfied; otherwise try again.
    UpdateContext(n_q, n_v, n_u, plant, context, plant_ad, context_ad,
                  projected_state);
    if (IsSampleWithinDistanceOfSurface(
      n_q, n_v, n_u, sampling_params.sample_projection_clearance,
      projected_state, plant, context, plant_ad, context_ad, contact_geoms,
      sampling_c3_options, min_distance_index)) {
      continue;
    }
    // Check the projection is within a small epsilon of the sampling height;
    // otherwise try again.
    // WARNING:  This assumes the walls of the object are roughly vertical.
    double epsilon = 0.001;
    if (projected_state[2] < sampling_params.sampling_height - epsilon ||
        projected_state[2] > sampling_params.sampling_height + epsilon) {
      continue;
    }

    // Undo the update context.
    UpdateContext(n_q, n_v, n_u, plant, context, plant_ad, context_ad, x_lcs);
    Eigen::Vector3d sample = projected_state.head(3);
    return sample;
  }
}
```

### `examples/sampling_c3/generate_samples.cc:403-441` — `IsSampleWithinDistanceOfSurface` (uses `GeomGeomCollider`)

```cpp
bool IsSampleWithinDistanceOfSurface(
    const int& n_q, const int& n_v, const int& n_u,
    const double& clearance_distance,
    const Eigen::VectorXd& candidate_state,
    drake::multibody::MultibodyPlant<double>& plant,
    drake::systems::Context<double>* context,
    drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
    drake::systems::Context<drake::AutoDiffXd>* context_ad,
    const std::vector<
      std::vector<drake::SortedPair<drake::geometry::GeometryId>>>&
      contact_geoms,
    SamplingC3Options sampling_c3_options,
    int& min_distance_index)
{
  // Update the context of the plant with the candidate state.
  UpdateContext(n_q, n_v, n_u, plant, context, plant_ad, context_ad,
                candidate_state);

  // Find the closest pair if there are multiple pairs
  std::vector<double> distances;
  for (int i = 0; i < contact_geoms.at(1).size(); i++) {
    SortedPair<GeometryId> pair{(contact_geoms.at(1)).at(i)};
    multibody::GeomGeomCollider collider(plant, pair);

    auto [phi_i, J_i] = collider.EvalPolytope(
      *context, sampling_c3_options.num_friction_directions);
    distances.push_back(phi_i);
  }

  // Find the minimum distance.
  auto min_distance_it = std::min_element(distances.begin(), distances.end());
  min_distance_index = std::distance(distances.begin(), min_distance_it);
  double min_distance = *min_distance_it;

  // Require that min_distance be at least 1 mm within the clearance distance.
  return min_distance <= clearance_distance - 1e-3;
}
```

### `examples/sampling_c3/generate_samples.cc:443-472` — `ProjectSampleOutsideObject` (push EE-witness-point outward by EE-radius + clearance)

```cpp
Eigen::VectorXd ProjectSampleOutsideObject(
    Eigen::VectorXd& candidate_state, int min_distance_index,
    const SamplingParams& sampling_params,
    const drake::multibody::MultibodyPlant<double>& plant,
    const drake::systems::Context<double>& context,
    const std::vector<
        std::vector<drake::SortedPair<drake::geometry::GeometryId>>>&
        contact_geoms) {

  // Compute the witness points between the penetrating sample and the object
  // surface.
  multibody::GeomGeomCollider collider(
    plant, contact_geoms.at(1).at(min_distance_index));
  auto [p_world_contact_ee, p_world_contact_obj] = collider.CalcWitnessPoints(
    context);

  // Get the EE radius to factor into the projection.
  double ee_radius = GetEERadiusFromPlant(plant, context, contact_geoms);

  // Find vector in direction from EE to object witness points.
  Eigen::Vector3d ee_to_obj = p_world_contact_obj - p_world_contact_ee;
  Eigen::Vector3d ee_to_obj_normalized = ee_to_obj.normalized();
  // Add clearance to the object in the same direction.
  Eigen::Vector3d p_world_contact_obj_clearance =
    p_world_contact_obj +
    (ee_radius + sampling_params.sample_projection_clearance) *
      ee_to_obj_normalized;
  candidate_state.head(3) = p_world_contact_obj_clearance;
  return candidate_state;
}
```

### `examples/sampling_c3/generate_samples.cc:364-378` — workspace test

```cpp
bool IsSampleInWorkspace(const Eigen::VectorXd& candidate_state,
                         const SamplingC3Options& sampling_c3_options) {
  double candidate_radius =
    sqrt(std::pow(candidate_state[0], 2) + std::pow(candidate_state[1], 2));
  if (candidate_state[0] < sampling_c3_options.workspace_limits[0][3] // x min
   || candidate_state[0] > sampling_c3_options.workspace_limits[0][4] // x max
   || candidate_state[1] < sampling_c3_options.workspace_limits[1][3] // y min
   || candidate_state[1] > sampling_c3_options.workspace_limits[1][4] // y max
   || candidate_state[2] < sampling_c3_options.workspace_limits[2][3] // z min
   || candidate_state[2] > sampling_c3_options.workspace_limits[2][4] // z max
   || candidate_radius > sampling_c3_options.robot_radius_limits[1]   // r min
   || candidate_radius < sampling_c3_options.robot_radius_limits[0])  // r max
   {return false;}
  return true;
}
```

---

## 8 — Per-loop sample generation in `SamplingC3Controller`

### `systems/controllers/sampling_based_c3_controller.cc:480-525` — every-tick sample generation + LCS-per-sample build

```cpp
  // Build C3Options from SamplingC3Options based on the
  // crossed_cost_switching_threshold_ flag.
  C3Options c3_options = sampling_c3_options_.GetC3Options(
    crossed_cost_switching_threshold_);

  // Update the cost matrices:  Q_, R_, G_, and U_.
  UpdateCostMatrices(x_lcs_curr, x_lcs_des, c3_options);

  // Generate states, differing from the current state only by EE sample
  // locations.
  std::vector<Eigen::VectorXd> candidate_states =
    GenerateSampleStates(n_q_, n_v_, n_u_, x_lcs_curr, is_doing_c3_,
                         sampling_params_, sampling_c3_options_, plant_,
                         context_, plant_ad_, context_ad_, contact_pairs_);

  // Add the previous best repositioning target to the candidate states at the
  // index 1 always. (Index 0 will become the current state.)
  if (!is_doing_c3_) {
    Eigen::VectorXd repositioning_target_state = x_lcs_curr;
    repositioning_target_state.head(3) = prev_repositioning_target_;
    candidate_states.insert(candidate_states.begin(),
                            repositioning_target_state);
  }
  // Insert the current location at the beginning of the candidate states.
  candidate_states.insert(candidate_states.begin(), x_lcs_curr);
  int num_total_samples = candidate_states.size();

  // Update the set of sample locations under consideration.
  all_sample_locations_.clear();
  for (int i = 0; i < num_total_samples; i++) {
    all_sample_locations_.push_back(candidate_states[i].head(3));
  }

  // Make LCS objects for each sample.
  auto lcs_pair = SamplingC3Controller::CreateLCSObjectsForSamples(
    candidate_states, x_lcs_curr, c3_options, c3_options);
  std::vector<solvers::LCS> lcs_candidates = lcs_pair.first;
  std::vector<solvers::LCS> lcs_candidates_for_cost = lcs_pair.second;
```

`SampleIndex::kCurrentLocation = 0`, `SampleIndex::kCurrentReposTarget = 1` (when in repos mode), others = additional samples.

---

## 9 — Sample-buffer maintenance (pose-pruning + add-new + lowest-cost-to-end)

### `systems/controllers/sampling_based_c3_controller.cc:1223-1335` — `MaintainSampleBuffer`

```cpp
void SamplingC3Controller::MaintainSampleBuffer(const VectorXd& x_lcs) const {
  // Determine if samples are outdated by comparing to the current object
  // position and orientation.
  Vector3d object_pos = x_lcs.segment(n_q_-3, 3);
  Eigen::Vector4d object_quat = x_lcs.segment(3, 4).normalized();

  MatrixXd buffer_xyzs =
    sample_buffer_.block(0, 7, sampling_params_.N_sample_buffer, 3);
  MatrixXd buffer_quats =
    sample_buffer_.block(0, 3, sampling_params_.N_sample_buffer, 4);

  // First, remove outdated samples that have moved too much from current object
  // configuration.
  VectorXd quat_dots = (buffer_quats * object_quat).array().abs();
  VectorXd angles = (2.0 * quat_dots.array().acos());
  Eigen::Array<bool, Eigen::Dynamic, 1> mask_satisfies_rot =
    (angles.array() < sampling_params_.ang_error_sample_retention);

  MatrixXd pos_deltas = buffer_xyzs.rowwise() - object_pos.transpose();
  VectorXd distances = pos_deltas.rowwise().norm();
  Eigen::Array<bool, Eigen::Dynamic, 1> mask_satisfies_pos =
    (distances.array() < sampling_params_.pos_error_sample_retention);

  MatrixXd retained_samples =
    MatrixXd::Zero(sampling_params_.N_sample_buffer, n_q_);
  VectorXd retained_costs =
    -1 * VectorXd::Ones(sampling_params_.N_sample_buffer);
  int retained_count = 0;
  for (int i = 0; i < sampling_params_.N_sample_buffer; i++) {
    if (mask_satisfies_rot[i] && mask_satisfies_pos[i]) {
      retained_samples.row(retained_count) = sample_buffer_.row(i);
      retained_costs[retained_count] = sample_costs_buffer_[i];
      retained_count++;
    } else if (sample_costs_buffer_[i] < 0) {
      break;
    }
  }
  sample_buffer_ = retained_samples;
  sample_costs_buffer_ = retained_costs;

  // Second, in preparation for adding new samples stored in
  // all_sample_locations_ (excluding the current location), if the buffer is
  // going to overflow, get rid of the oldest samples first.  NOTE:  Step 4
  // moves the lowest cost sample in the buffer to the end, so the best sample
  // is usually excluded from this cut.
  int num_to_add = all_sample_locations_.size() - 1;
  if (!is_doing_c3_) {
    num_to_add--;
  }
  if (retained_count + num_to_add > sampling_params_.N_sample_buffer) {
    int shift_by =
      retained_count + num_to_add - sampling_params_.N_sample_buffer;
    retained_count -= shift_by;
    sample_buffer_.block(0, 0, retained_count, n_q_) =
      sample_buffer_.block(shift_by, 0, retained_count, n_q_);
    sample_costs_buffer_.segment(0, retained_count) =
      sample_costs_buffer_.segment(shift_by, retained_count);
  }

  // Third, add the new samples stored in all_sample_locations_ and
  // all_sample_costs_.  Don't add the current location (so the sample buffer
  // contains more broadly sampled locations) or a currently pursued
  // repositioning target.
  int buffer_count = retained_count;
  for (int i = retained_count;
       i < retained_count + all_sample_locations_.size(); i++) {
    DRAKE_DEMAND(buffer_count < sampling_params_.N_sample_buffer);
    if ((i == retained_count) || (!is_doing_c3_ && i == retained_count + 1)) {
      // Skip the current location.
      // Skip the repositioning target if in repositioning mode.
    } else {
      VectorXd new_config = x_lcs.segment(0, n_q_);
      new_config.segment(0, 3) = all_sample_locations_[i - retained_count];
      // Ensure a normalized quaternion is written to the buffer.
      new_config.segment(3, 4) = object_quat;
      sample_buffer_.row(buffer_count) = new_config;
      sample_costs_buffer_[buffer_count] =
        all_sample_costs_[i - retained_count];
      buffer_count++;
    }
  }
  num_in_buffer_ = buffer_count;

  // Lastly, ensure the lowest cost sample is at the end of the buffer.
  VectorXd eligible_costs = sample_costs_buffer_.head(num_in_buffer_);
  int lowest_cost_index;
  double lowest_buffer_cost = eligible_costs.minCoeff(&lowest_cost_index);
  VectorXd lowest_cost_sample = sample_buffer_.row(lowest_cost_index);
  sample_buffer_.row(lowest_cost_index) =
    sample_buffer_.row(num_in_buffer_ - 1);
  sample_costs_buffer_[lowest_cost_index] =
    sample_costs_buffer_[num_in_buffer_ - 1];
  sample_buffer_.row(num_in_buffer_ - 1) = lowest_cost_sample;
  sample_costs_buffer_[num_in_buffer_ - 1] = lowest_buffer_cost;

  DRAKE_DEMAND(sample_buffer_.rows() == sampling_params_.N_sample_buffer);
  DRAKE_DEMAND(sample_buffer_.cols() == n_q_);
  DRAKE_DEMAND(sample_costs_buffer_.size() == sampling_params_.N_sample_buffer);
}
```

Three phases per loop:
1. **Prune** outdated samples (`||obj_pos − sample.obj_pos|| > pos_error_sample_retention` OR geodesic angle `> ang_error_sample_retention`).
2. **Add new** samples from `all_sample_locations_`, skipping the current position (and the repos target when in repos mode). FIFO-evict if overflow, with the lowest-cost sample protected by being at the buffer end.
3. **Reorder** so lowest-cost is at index `num_in_buffer_ − 1`.

---

## 10 — Mode-switch hysteresis (c3 ↔ repos)

### `systems/controllers/sampling_based_c3_controller.cc:620-770` — six hysteresis knobs + decision

```cpp
  // Set up hysteresis values based on if the cost switching threshold has been
  // crossed.
  double hyst_c3_to_repos = progress_params_.hyst_c3_to_repos;
  double hyst_repos_to_c3 = progress_params_.hyst_repos_to_c3;
  double hyst_repos_to_repos = progress_params_.hyst_repos_to_repos;
  double hyst_c3_to_repos_frac = progress_params_.hyst_c3_to_repos_frac;
  double hyst_repos_to_c3_frac = progress_params_.hyst_repos_to_c3_frac;
  double hyst_repos_to_repos_frac = progress_params_.hyst_repos_to_repos_frac;
  if (!crossed_cost_switching_threshold_) {
    hyst_c3_to_repos = progress_params_.hyst_c3_to_repos_position;
    hyst_repos_to_c3 = progress_params_.hyst_repos_to_c3_position;
    hyst_repos_to_repos = progress_params_.hyst_repos_to_repos_position;
    hyst_c3_to_repos_frac = progress_params_.hyst_c3_to_repos_frac_position;
    hyst_repos_to_c3_frac = progress_params_.hyst_repos_to_c3_frac_position;
    hyst_repos_to_repos_frac =
      progress_params_.hyst_repos_to_repos_frac_position;
  }

  // Review the cost results to determine the best sample.
  bool force_c3_mode = radio_out->channel[12];
  double best_other_cost;
  if (num_total_samples > 1) {
    std::vector<double> additional_sample_cost_vector = std::vector<double>(
      all_sample_costs_.begin() + 1, all_sample_costs_.end());
    best_other_cost = *std::min_element(additional_sample_cost_vector.begin(),
                                        additional_sample_cost_vector.end());
    …
    best_sample_index_ = (SampleIndex)(
      std::distance(std::begin(additional_sample_cost_vector), it) + 1);
  } else {
    force_c3_mode = true;
  }

  // Determine whether to do C3 or reposition.
  mode_switch_reason_ = ModeSwitchReason::kNoSwitch;
  double curr_cost = all_sample_costs_[SampleIndex::kCurrentLocation];
  double repos_target_cost =all_sample_costs_[SampleIndex::kCurrentReposTarget];
  if (is_doing_c3_ == true) {  // Currently doing C3.
    pursued_target_source_ = PursuedTargetSource::kNoTarget;

    // Keep track of progress while in C3 mode.
    bool met_minimum_progress = true;  // Reset by below function.
    bool print_current_pos_and_rot_cost = radio_out->channel[6];
    KeepTrackOfC3ModeProgress(
      x_lcs_curr, x_lcs_final_des, met_minimum_progress,
      print_current_pos_and_rot_cost);

    // Switch to repositioning if progress was insufficient.
    if (!met_minimum_progress && !force_c3_mode &&
        (sampling_params_.num_additional_samples_c3 > 0)) {
      is_doing_c3_ = false;
      mode_switch_reason_ = ModeSwitchReason::kToReposUnproductive;
      std::cout << "Repositioning after not making progress in C3" << std::endl;
    }

    // Switch to repositioning if one of the other samples is better, with
    // hysteresis.
    else if (
      ((!progress_params_.use_relative_hysteresis &&
        curr_cost > best_other_cost + hyst_c3_to_repos) ||
       (progress_params_.use_relative_hysteresis &&
        curr_cost > best_other_cost * (1 + hyst_c3_to_repos_frac))) &&
      !force_c3_mode &&
      (sampling_params_.num_additional_samples_c3 > 0)) {
      is_doing_c3_ = false;
      mode_switch_reason_ = ModeSwitchReason::kToReposCost;
      …
    }
    …
```

Three hysteresis tiers, each available in **absolute** (`hyst_X_to_Y`) and **relative-frac** (`hyst_X_to_Y_frac`) variants:
- `c3_to_repos` — leave c3 when `curr_cost > best_other_cost + hyst`
- `repos_to_c3` — re-enter c3 from repos when `best_other_cost > curr_cost + hyst`
- `repos_to_repos` — switch to a better repos target within repos mode

Two regimes (`crossed_cost_switching_threshold_` flag) — pre-threshold (`*_position` variants) and post-threshold use different hysteresis values. Selected via `use_relative_hysteresis` toggle.

`ModeSwitchReason` values: `kNoSwitch`, `kToReposUnproductive` (progress timeout), `kToReposCost` (cost gap), `kToC3Cost`, `kToC3ReachedReposTarget`.

---

## 11 — README highlights

### `examples/sampling_c3/README.md:1-30`

```
# Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity
This is an implementation of our paper currently available on Arxiv.

[[Project webpage](https://approximating-global-ci-mpc.github.io/)]
[[Arxiv](https://arxiv.org/abs/2505.13350)]
[[Supplemental video](https://youtu.be/rv9n8Uyvoh0)]

## Simulation Experiments

1. Start the procman script …
2. In the procman window, start the meshcat visualizer …
3. The examples with the sampling C3 controller can be run using the script
   `script:start_experiment_no_logs`. Scripts are located in the top bar of
   the procman window. This script spawns three processes:
   - `bazel-bin/examples/sampling_c3/franka_sim`: Simulated environment which
     takes in torques commands from `franka_osc` and publishes the state of
     the system via LCM on various channels.
   - `bazel-bin/examples/sampling_c3/franka_osc_controller`: Low-level
     task-space controller that tracks task-space trajectories it receives
     from the MPC.
   - `bazel-bin/examples/sampling_c3/franka_sampling_c3_controller`: Contact
     Implicit MPC controller that takes in the state of the system and
     publishes end effector trajectories to be tracked by the OSC.
```

Three-process pipeline:
- `franka_sim` — Drake sim, publishes state
- `franka_sampling_c3_controller` — C3 MPC, **plans + publishes EE trajectories** (position, orientation, force)
- `franka_osc_controller` — OSC, **subscribes to trajectories + outputs joint torques**

LCM channels (in `parameter_headers/lcm_channels.h`):
- `tracking_trajectory_actor_channel` — bundle containing `end_effector_position_target` + `end_effector_orientation_target` + `end_effector_force_target`
- `franka_state_channel` — state out from sim
- `franka_input_channel` — torque commands in
- `osc_debug_channel` — `lcmt_osc_output` debug

---

## Reference info

| Item | Value |
|---|---|
| Repo | `https://github.com/DAIRLab/dairlib.git` |
| Branch | `sampling_based_c3_public` |
| HEAD | `b52c68d Fix error in quaternion error hessian (fraction inadvertently converted to integer)` |
| Paper | arXiv 2505.13350 — Venkatesh, Bianchini, Aydinoglu, Yang, Posa (2025) |
| Local clone | `/root/reference_repos/dairlib_sampling_c3/` |
| Local mirror | `/d/projects/ERL/reference_repos/dairlib_sampling_c3/` |

`AddFrankaToPlant` / `AddObjectToPlant` / `AddLCSModelsToPlant` are defined in `examples/sampling_c3/sampling_c3_utils.cc:14-87` and declared in `sampling_c3_utils.h`. The full file is short (89 lines) and reproduced inline in §1.
