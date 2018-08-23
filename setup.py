from __future__ import absolute_import
from __future__ import print_function

import os

from setuptools import find_packages
from setuptools import setup

print("Generating estdel/__version__.py: ", end="")
__version__ = open("VERSION").read().strip()
print(__version__)
open("estdel/__version__.py", "w").write('__version__="%s"' % __version__)

# read the latest git status out to an installed file
try:

    gitbranch = os.popen("git symbolic-ref -q HEAD").read().strip()
    print("Generating estdel/__branch__.py")

    gitlog = os.popen('git log -n1 --pretty="%h%n%s%n--%n%an%n%ae%n%ai"').read().strip()
    print("Generating estdel/__gitlog__.py.")
    print(gitlog)

except:
    gitbranch = "unknown branch"
    gitlog = "git log not found"

open("estdel/__branch__.py", "w").write('__branch__ = "%s"' % gitbranch)
open("estdel/__gitlog__.py", "w").write('__gitlog__ = """%s"""' % gitlog)


def get_description():
    def get_description_lines():
        seen_desc = False

        with open('README.md') as f:
            for line in f:
                if seen_desc:
                    if line.startswith('##'):
                        break
                    line = line.strip()
                    if line:
                        yield line
                elif line.startswith('## Description'):
                    seen_desc = True

    return ' '.join(get_description_lines())


setup(
    name="estdel",
    version=__version__,
    description="Estimate interferometer anteanna delays",
    long_description=get_description(),
    url="https://github.com/andrewasheridan/estimating_delays/estdel",
    author="Andrew Sheridan",
    author_email="sheridan@berkeley.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=1.8.0",
        "numpy!=1.15.0",
    ],
    zip_safe=False,
    include_package_data=True,
    test_suite="estdel.tests",
)

# ???: On setup there are many 'Adding xxx to easy-install.pth file'. Why?
