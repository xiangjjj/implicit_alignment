import warnings


def filter_deprecation_warning():
    print('Caution: deprecation warnings are filtered!')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
