
from run import main


def run_main_with_command_line_args(main_function, **kwargs):
    """ Runs `main` with the given command line arguments passed as kwargs.
    """
    main_args = []
    for key in kwargs:
        main_args.extend([f"--{key}", str(kwargs[key])])
    main_function.main(main_args)