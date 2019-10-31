from setuptools import setup, find_packages

try:
    import commit

    commit_version = commit.___version__.split('.')
    commit_version = [int(version_val) for version_val in commit_version]
    if commit_version[0] == 1 and commit_version[1] < 3:
        raise RuntimeError( 'AMICO is not compatible with the current version of COMMIT.\nPlease do the following:\n1. Uninstall COMMIT.\n2. Try to re-install AMICO.\n3. Install COMMIT v1.3.0 or above.' )
except:
    print('NOTE: If you are going to install COMMIT, make sure you install COMMIT v1.3.0 or above.')

setup(
    name='amico',
    version='1.1.0',
    description='Accelerated Microstructure Imaging via Convex Optimization (AMICO)',
    author='Alessandro Daducci',
    author_email='alessandro.daducci@gmail.com',
    url='https://github.com/daducci/AMICO',
    packages=find_packages(),
    package_data={'' : ['*.bin', 'directions/*.bin']},
    install_requires=[
        'dipy',
    ],
)
