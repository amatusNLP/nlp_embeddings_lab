from setuptools import setup, find_packages


setup(
      name='nlp_embeddings_lab',
      version='0.1',
      description='An interface to various nlp embeddings tools',
      url='#',
      author='Nicola Cirillo',
      author_email='nicola.cirillo96@outlook.it',
      license='MIT',
      packages=find_packages(),
      package_data = {'':['ALaCarte/**/*']},
      include_package_data=True,
      zip_safe=False,
      install_requires=[
              'gensim',
              'scikit-learn',
              'chars2vec',
              'conll_iterator',
              'tqdm',
              'keras',
              'tensorflow',
              ],
        dependency_links = ['https://github.com/IntuitionEngineeringTeam/chars2vec']
      )

