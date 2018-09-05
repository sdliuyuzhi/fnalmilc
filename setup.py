from setuptools import setup, find_packages

requires = [
    'numpy',
    'gvar',
    'lsqfit',
    'tabulate',
    'argparse',
    'nose'
]

def readme():
    with open('README.rst') as f:
        return f.read()

#if sys.version_info < (3, 2):
#    requires.append('futures==2.2')

setup(name='fnalmilc',
    version='0.1',
    description='fnalmilc is an application written in Python that'
    'construct the phenomenology observables using the Bs -> K l nu '
    'form factors, f+ and f0.',
    #url='http://github.com/',
    author='Yuzhi Liu',
    author_email='liuyuzhi83@gmail.com',
    license='BSD 2-Clause License',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    #packages=['funniest'],
    test_suite='nose.collector',
    install_requires=requires,
    entry_points={
        'console_scripts': ['fnalmilc=fnalmilc.app:main'],
    },
    zip_safe=False,
    keywords=['form factor', 'semileptonic', 'Bs2K', 'FNAL', 'MILC',
        'Fermilab', 'Lattice', 'QCD', 'phenomenology', 'decay rate']
)

