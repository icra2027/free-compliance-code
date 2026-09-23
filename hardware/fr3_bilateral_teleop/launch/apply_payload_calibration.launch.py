#  Copyright (c) 2026 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Apply a previously-run payload calibration (calibrate_payload.py) to an already-running
follower via the set_load service, WITHOUT bringing up the full leader+follower teleop.launch.py
pair.

Fills a real gap: calibrate_payload.launch.py deliberately does NOT call set_load (its whole
precondition is running BEFORE the new payload is configured -- see calibrate_payload.py's
docstring), and teleop.launch.py DOES call set_load but only as part of standing up a full
leader+follower pair. Anything that needs set_load applied to a follower-only bringup (e.g.
collect_free_space_sweep.py, whose own precondition is "run this with the follower's normal
set_load payload already configured") had no way to get that without either hand-rolling the
`ros2 service call ... set_load` command or connecting a leader it doesn't need.

Reads <namespace>_payload.yaml the same way teleop.launch.py's load_payload_calibration()/
create_set_load_process() do (duplicated here rather than imported -- these are plain launch
Python files, not an installed importable package module, so duplication is the pragmatic
choice over a larger packaging refactor). Missing/failed calibration is a loud warning and a
zero-load set_load call, matching teleop.launch.py's own fallback behavior, not a launch failure.

Usage (after calibrate_payload.launch.py has brought up the follower and
franka_robot_state_broadcaster/controller_manager services are up):
    ros2 launch fr3_bilateral_teleop apply_payload_calibration.launch.py \
        namespace:=franka_teleop/follower
"""
import os
from typing import Any, Dict

import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import FindExecutable, LaunchConfiguration

PAYLOAD_OUTPUT_DIR_ENV = "TELEOP_PAYLOAD_OUTPUT_DIR"
PAYLOAD_OUTPUT_DIR_DEFAULT = "/tmp/franka_teleop_payload"


def cvt_to_string(values):
    return ", ".join(str(float(v)) for v in values)


def default_set_load_values() -> Dict[str, Any]:
    return {"mass": 0.0, "center_of_mass": [0.0, 0.0, 0.0], "load_inertia": [0.0] * 9}


def load_payload_calibration(namespace: str) -> Dict[str, Any]:
    output_dir = os.environ.get(PAYLOAD_OUTPUT_DIR_ENV, PAYLOAD_OUTPUT_DIR_DEFAULT)
    file_namespace = namespace.strip("/").replace("/", "_")
    payload_path = os.path.join(output_dir, f"{file_namespace}_payload.yaml")

    if not os.path.exists(payload_path):
        print(
            f"[apply_payload_calibration.launch] WARNING: no payload calibration file at "
            f"'{payload_path}' for '{namespace}'. Run calibrate_payload.launch.py + "
            f"calibrate_payload.py first. Falling back to a zero/no-extra-load default -- "
            f"anything relying on set_load being correct (e.g. a free-space bias sweep) will "
            f"see a phantom bias from the undeclared payload instead of real data.")
        return default_set_load_values()

    try:
        with open(payload_path, "r") as payload_file:
            report = yaml.safe_load(payload_file)
        if report.get("status") != "ok" or "set_load" not in report:
            print(
                f"[apply_payload_calibration.launch] WARNING: payload calibration at "
                f"'{payload_path}' has status={report.get('status')!r} (not 'ok') -- falling "
                f"back to a zero/no-extra-load default for '{namespace}'.")
            return default_set_load_values()
        return report["set_load"]
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        print(
            f"[apply_payload_calibration.launch] WARNING: failed to read payload calibration "
            f"at '{payload_path}': {exc}. Falling back to a zero/no-extra-load default for "
            f"'{namespace}'.")
        return default_set_load_values()


def launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration("namespace").perform(context)
    set_load_values = load_payload_calibration(namespace)
    print(
        f"[apply_payload_calibration.launch] Applying to '{namespace}': "
        f"mass={set_load_values['mass']}, "
        f"center_of_mass={set_load_values['center_of_mass']}")
    return [
        ExecuteProcess(
            cmd=[[
                FindExecutable(name="ros2"),
                " service call ",
                f"/{namespace}/service_server/set_load ",
                "franka_msgs/srv/SetLoad ",
                "\"{ ",
                f"mass: {float(set_load_values['mass'])}, ",
                f"center_of_mass: [{cvt_to_string(set_load_values['center_of_mass'])}], ",
                f"load_inertia: [{cvt_to_string(set_load_values['load_inertia'])}] ",
                "}\"",
            ]],
            shell=True,
            name="set_robot_payload",
            output="both",
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "namespace", default_value="follower",
            description="Namespace of the already-running arm to apply set_load to"),
        OpaqueFunction(function=launch_setup),
    ])
