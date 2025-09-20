from setuptools import setup

setup(
    name='jmrespy',
    version='1.5.0',    
    description='A package which fit joint models for longitudinal and time-to-event data whith shared random effects. And make survival prgniostics from them',
    author='Lucas Chabeau, Sêmeia, INSER - UMR 1246 - SPHERE',
    author_email='lucas.chabeau@gmail.com',
    packages=['jmrespy', 'jmrespy.utils'],
    install_requires=[
        'numpy==1.24.2',
        'pandas==1.4.1',
        'statsmodels==0.13.2',
        'lifelines==0.27.4',
        'patsy==0.5.2',
        'scipy==1.8.0',
        'tabulate==0.8.9',
        'tqdm==4.63.0',
        'bambi==0.10.0',
        'scikit-learn==1.5.0',
        'joblib==1.2.0',
        'mpire==2.7.1'
    ]

)