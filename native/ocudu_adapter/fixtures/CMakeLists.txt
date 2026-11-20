# Sprint 2.1: transmit reference of one grant, built with the pinned test's transmit chain.
add_executable(grant_fixture grant_fixture.cpp)
target_link_libraries(grant_fixture PRIVATE ocudu_pin_fixture)

# The executable starts and names its options. A missing argument is refused with status 2.
add_test(NAME grant_fixture_help COMMAND grant_fixture --help)
set_tests_properties(grant_fixture_help PROPERTIES PASS_REGULAR_EXPRESSION "--dmrs-reference crb0")
add_test(NAME grant_fixture_refuses_missing_arguments COMMAND grant_fixture --no-loopback)
set_tests_properties(grant_fixture_refuses_missing_arguments PROPERTIES WILL_FAIL TRUE)
