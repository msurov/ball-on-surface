import tests.ball_on_rotary_cone_dynamics_test
import tests.ball_on_rotary_plane_dynamics_test
import tests.rigid_body_dynamics_test
import tests.rotating_coordinate_system_test

def run_all_tests():
  tests.ball_on_rotary_cone_dynamics_test.test()
  tests.ball_on_rotary_plane_dynamics_test.test()
  tests.rigid_body_dynamics_test.test()
  tests.rotating_coordinate_system_test.test()

run_all_tests()
