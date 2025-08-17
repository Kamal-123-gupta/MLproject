from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'#for ignoring this -e .

def get_requirements(file_path:str)->list[str]:
    '''
    this function will read requirements from requirements.txt

    '''

    requirements = []#get libraries to this list
    with open(file_path) as file_object:
        requirements = file_object.readlines()
        requirements=[req.replace('\n',"") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
    name='MLproject',
    version='0.0.1',
    author='kamal',
    author_email='kg381852@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)