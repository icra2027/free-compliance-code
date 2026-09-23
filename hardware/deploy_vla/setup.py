from setuptools import find_packages, setup

package_name = 'deploy_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anonymous',
    maintainer_email='anonymous@example.invalid',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'deploy_vla = deploy_vla.deploy_vla:main',
            'hand_eye_calibration = deploy_vla.hand_eye_calibration:main',
            'deploy_vla_crisp = deploy_vla.deploy_vla_crisp:main',
            'deploy_vla_cic = deploy_vla.deploy_vla_cic:main',
            'deploy_smolvla = deploy_vla.deploy_smolvla:main',
            'run_scripted_rollout = deploy_vla.run_scripted_rollout:main',
            'run_final_evaluation = deploy_vla.run_final_evaluation:main',
            'score_wipe = deploy_vla.score_wipe:main',
        ],
    },
)
