#! python

import os
import os.path as osp
import sys

ROOT_DIR = osp.abspath(osp.dirname(__file__))
SOURCE_DIR = osp.join(ROOT_DIR, 'source')

if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

cur_pythonpath = os.environ.get('PYTHONPATH', '')
if cur_pythonpath:
    if SOURCE_DIR not in cur_pythonpath.split(os.pathsep):
        os.environ['PYTHONPATH'] = SOURCE_DIR + os.pathsep + cur_pythonpath
else:
    os.environ['PYTHONPATH'] = SOURCE_DIR

from arkvale.chat_arkvale import ArkValeChatBot 

if __name__ == '__main__':
    from opencompass.cli.main import main

    main()
