import os

def _main(background):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    blender = os.environ['blender']
    args.args[0] = '"{}"'.format(args.args[0])
    cmd = f'{blender} -noaudio {"-b" if background else ""} --python {args.path} -- {" ".join(args.args)}'
    print(cmd)
    os.system(cmd)

def back():
    _main(background=True)

def main():
    _main(background=False)


if __name__ == '__main__':
    _main()