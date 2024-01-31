__version__ = "0.9.9"


def get_version():
    return __version__


def get_name_version():
    return f"optboolnet-{__version__}"


if __name__ == "__main__":
    import sys

    args = sys.argv[1]
    if args == "name":
        print(get_name_version())
    elif args == "version":
        print(get_version())
