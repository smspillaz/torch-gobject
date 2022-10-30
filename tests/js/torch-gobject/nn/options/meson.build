torch_gobject_nn_options_js_tests = [
  'testGeneratedOptions.js'
]

if jasmine.found()
  foreach test_file : torch_gobject_nn_options_js_tests
    test(test_file,
         gjs,
         args: [
           jasmine.full_path(),
           '--verbose',
           join_paths(meson.current_source_dir(), test_file)
         ],
         env: tests_environment,
         depends: [torch_gobject_tests_resources_typelib])
  endforeach
endif