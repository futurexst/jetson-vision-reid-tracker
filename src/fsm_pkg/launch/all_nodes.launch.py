from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="fsm_pkg",
            executable="fsm_node",
            name="fsm_node",
            output="screen",
        ),
        Node(
            package="follow_tracker",
            executable="follow_node",
            name="follow_node",
            output="screen",
        ),
        Node(
            package="tts_pkg",
            executable="tts_node",
            name="tts_node",
            output="screen",
        ),
        Node(
            package="pickup_pkg",
            executable="pickup_node",
            name="pickup_node",
            output="screen",
        ),
    ])
