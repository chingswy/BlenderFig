'''
  @ Date: 2021-03-02 16:53:55
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-14 16:20:10
  @ FilePath: /EasyMocap/setup.py
'''
from setuptools import setup

setup(
    name='myblender',     
    version='0.1',   #
    description='Blender Toolbox',
    author='Qing Shuai', 
    author_email='s_q@zju.edu.cn',
    # test_suite='setup.test_all',
    packages=[
        'myblender',
    ],
    entry_points={
        'console_scripts': [
            'blenderqing=myblender.entry:main',
            'blenderqingback=myblender.entry:back',
        ],
    },
    install_requires=[],
    data_files = []
)
