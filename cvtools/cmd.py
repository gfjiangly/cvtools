# -*- encoding:utf-8 -*-
# @Time    : 2020/2/11 11:45
# @Author  : jiang.g.f
# @File    : cmd.py
# @Software: PyCharm

import os
import os.path as osp
import sys
import argparse
import importlib

import cvtools

COLOR = True


def write(text, color=None):
    global COLOR
    if COLOR:
        text = text.replace('==>', '\033[90m==>\033[0m')
        text = text.replace('    +', '    \033[32m+\033[0m')
        text = text.replace('XX>', '\033[31mXX>\033[0m')
        if text[:6] == 'Error:':
            text = '\033[41mError:\033[0m\033[91m%s\033[0m' % text[6:]
        elif text[:4] == 'Tip:':
            text = '\033[42mTip:\033[0m\033[32m%s\033[0m' % text[4:]
        elif text.strip()[:4] == 'http':
            text = '\033[92m%s\033[0m' % text
        elif text[:7] == 'CVTools':
            text = """
    _________ ____   _______________              .__           
    \_   ___ \\   \ /   /\__    ___/____    ____  |  |    ______
    /    \  \/ \   Y   /   |    |  /  _ \  /  _ \ |  |   /  ___/
    \     \____ \     /    |    | (  <_> )(  <_> )|  |__ \___ \ 
     \______  /  \___/     |____|  \____/  \____/ |____//____  >
            \/                                               \/ 
                                                          %s\n""" \
                   % ("v"+cvtools.__version__)
        elif color == 'red':
            text = '\033[91m%s\033[0m' % text
        elif color == 'green':
            text = '\033[92m%s\033[0m' % text

    sys.stdout.write(text + '\n')


version = "CVTools py-v{} - https://cvtools.readthedocs.io".format(
    cvtools.__version__)


def main(*argv):
    root = os.getcwd()

    # Build Parser
    # ------------
    parser = argparse.ArgumentParser(
        prog='cvtools', add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Upload reports to Codecov"""
    )
    basics = parser.add_argument_group(
        '======================== Basics ========================')
    basics.add_argument('--version', action='version', version=version)

    basics.add_argument('--deploy', '-d', default=None,
                        help="deploy a model in web")
    basics.add_argument('--port', '-p', default=5000, type=int,
                        help="model deployment port")
    basics.add_argument('--log', '-l', default="deploy/deploy.log",
                        help="dump logs to file")

    debugging = parser.add_argument_group(
        '======================== Debugging ========================')
    debugging.add_argument('--dump', action="store_true",
                           help="Dump collected data and do not send to CVTools")
    debugging.add_argument('-v', '--verbose', action="store_true",
                           help="Be verbose, e.g. dump the collected data")
    debugging.add_argument('--no-color', action="store_true",
                           help="Do not output with color")

    # Parse Arguments
    # ---------------
    if argv:
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()

    global COLOR
    COLOR = not args.no_color
    write('CVTools')

    if args.deploy:
        # sys.path.append(root)
        # deploy = importlib.import_module(args.deploy, package='.')
        with open(args.deploy, 'r') as f:
            exec(compile(f.read(), args.deploy, 'exec'))
        model_r = locals()['model']
        cvtools.run_model(model_r, args.port, args.log)


if __name__ == '__main__':
    main()
