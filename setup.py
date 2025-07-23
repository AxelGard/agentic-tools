from setuptools import setup, find_packages

setup(
    name='agentic-tools', 
    version='0.1.0',
    author='Axel Gard',
    author_email='axel.gard@tutanota.com',
    description='A toolset for agentic workflows',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AxelGard/agentic-tools',  
    packages=find_packages(include=['agentic_tools', 'agentic_tools.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        "langchain-ollama",
        "langchain",
    ],
    extras_require={
        'dev': [
            'pytest',
            "black",
            "build",
        ]
    },
    include_package_data=True,
)
