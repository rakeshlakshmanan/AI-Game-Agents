from setuptools import setup, find_packages

setup(
    name='ai-game-agents',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'torch>=1.9.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
        'tqdm>=4.62.0',
    ],
    python_requires='>=3.8',
    description='AI Game Agents: Minimax and RL for Tic Tac Toe and Connect 4',
    author='CS7IS2 Student',
)
