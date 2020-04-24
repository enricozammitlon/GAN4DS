# MPhys Project
For all documentation please check the Wiki tab on Github.

Current Status of deployment (Green means the job has been dispatched to hepgpu5): ![Run HepGPU](https://github.com/enricozammitlon/mphys-project/workflows/Run%20HepGPU/badge.svg?branch=deploys)
Is the run on hepgpu5 ready? (Red means it is still running):![Check Status](https://github.com/enricozammitlon/mphys-project/workflows/Check%20Status/badge.svg?branch=master&event=schedule)

Later on include this in the actions github

  validating-yaml-files:
    runs-on: [ubuntu-latest]
    steps:
    - uses: actions/checkout@v2

    - name: Setup Python environment
      uses: actions/setup-python@v1.1.1

    - name: Install pykwalify
      run: |
        pip3 install pykwalify PyYAML
    - name: General Config Validation
      run: |
        cd Standalone/tests
        pwd
        python3 config_test.py
    - name: Discriminator Layout Validation
      run: |
        cd Standalone/tests
        pwd
        python3 discriminator_test.py
        python3 config_test.py
        python3 generator_test.py