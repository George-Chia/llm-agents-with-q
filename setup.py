import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


setuptools.setup(
    name='llm-agent-with-q',
    description='Official Implementation of "Enhancing Decision-Making for LLM Agents via Step-Level Q-Value Models"',
    keywords='tree-search, direct policy optimization, large-language-models, llm, prompting, decision-making, reasoning, language-agent-tree-search',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    install_requires=[
        'setuptools',
    ],
    include_package_data=True,
)
