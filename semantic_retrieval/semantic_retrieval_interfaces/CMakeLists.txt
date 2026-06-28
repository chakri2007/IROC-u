cmake_minimum_required(VERSION 3.8)
project(semantic_retrieval_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SemanticMatch.msg"
  "srv/TriggerIndexing.srv"
  "srv/GetHdFrame.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_package()
