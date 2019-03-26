% Copy this template configuration file to your VOT workspace.
% Enter the full path to the ECO repository root folder.

ECO_repo_path = 'E:/ForwardBackward/Tracker/FBT_ECO';

tracker_label = 'ECO';
tracker_command = generate_matlab_command('benchmark_tracker_wrapper(''ECO'', ''eco_vot_deep'', true)', {[ECO_repo_path '/VOT_integration/benchmark_wrapper']});
tracker_interpreter = 'matlab';