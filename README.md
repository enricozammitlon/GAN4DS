# MPhys Project
For all documentation please check the Wiki tab on Github.

To start the training runner using docker:
* `git clone` this repository
* `docker pull enricozl/gan4ds:argan-runnerr`
* `docker run --mount type=bind,source=<Insert here absolute path>/GAN4DS/Training,target=/Training -p 0.0.0.0:6006:6006 enricozl/gan4ds:argan-runner`

To start developing using docker:
* Download VS Code
* Download the ms-vscode-remote.remote-containers extension
* From the extension tab (or by shift+cmd P) click `Open folder in container`
* Find the GAN4DS repistory folder
* Wait for everything to laod and you can start developing