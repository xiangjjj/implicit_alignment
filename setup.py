import os
from setuptools import setup

version = os.environ.get('VERSION', '0.0.0')

setup(name='ai.domain_adaptation',
      version='0.1',
      description='implicit class-conditioned domain alignment',
      author='Xiang',
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'torch>=1.4.0', 'easydict', 'pyyaml', 'tensorboardX', 'tqdm', 'torchvision', 'scikit-learn', 'numpy'
      ],
      entry_points={
          'console_scripts': [
              'implicit_alignment = ai.domain_adaptation.main:run']
      }
)
