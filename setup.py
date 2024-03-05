from setuptools import setup, find_packages
from typing import List

SETUP_FILE_EXCEPTION = '-e .'

def get_requirements(file_path: str) -> List[str]:
    all_requirements = []
    with open(file_path, 'r') as file:
        all_requirements = file.readlines()
        all_requirements = [requirement.strip() for requirement in all_requirements]
    
        if (SETUP_FILE_EXCEPTION in all_requirements):
            all_requirements.remove(SETUP_FILE_EXCEPTION)
    return all_requirements


setup(
    name='student_performance_indicator',
    version='0.0.1',
    author='Kaloyan Anastasov',
    author_email='test_email@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)