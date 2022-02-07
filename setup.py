from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='EntityEmbedding',
  version='0.1',
  description='Entity Embedding for Categorical Features',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Ilia Latauri, Avto Chakhnashvili',
  author_email='ilia.latauri@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='entityembedding, embedding', 
  packages=find_packages(),
  install_requires=[''] 
)