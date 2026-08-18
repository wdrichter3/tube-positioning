cmake_minimum_required(VERSION 3.8)
project(tube_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TubeCandidate.msg"
  "msg/Tube.msg"
  "msg/TubeMap.msg"
  "msg/CalibrationState.msg"
  "msg/WorkStep.msg"
  "msg/WorkPlan.msg"
  "msg/GuidanceTarget.msg"
  "srv/BuildTubeMap.srv"
  "srv/GeneratePlan.srv"
  DEPENDENCIES std_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
