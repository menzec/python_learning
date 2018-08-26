# -*-coding: utf-8 -*-
import platform
import pip
# pip V10.0.0以上版本需要导入下面的包
from pip._internal.utils.misc import get_installed_distributions
from subprocess import call
from time import sleep
import subprocess


def platform_info():
    # ouput system type and version info
    print("platform.machine()=%s", platform.machine())
    print("platform.node()=%s", platform.node())
    print("platform.platform()=%s", platform.platform())
    print("platform.processor()=%s", platform.processor())
    print("platform.python_build()=%s", platform.python_build())
    print("platform.python_compiler()=%s", platform.python_compiler())
    print("platform.python_branch()=%s", platform.python_branch())
    print("platform.python_implementation()=%s",
          platform.python_implementation())
    print("platform.python_revision()=%s", platform.python_revision())
    print("platform.python_version()=%s", platform.python_version())
    print("platform.python_version_tuple()=%s",
          platform.python_version_tuple())
    print("platform.release()=%s", platform.release())
    print("platform.system()=%s", platform.system())
    #print("platform.system_alias()=%s", platform.system_alias());
    print("platform.version()=%s", platform.version())
    print("platform.uname()=%s", platform.uname())


def pip_upgrade_package():
    if str(platform.system()) == 'Windows':
        for dist in get_installed_distributions():
            # 执行后，pip默认为Python3版本
            # 双版本下需要更新Python2版本的包，使用py2运行，并将pip修改成pip2
            #call("sudo pip install --upgrade " + dist.project_name, shell=True)
            call("pip install --upgrade " + dist.project_name, shell=True)
    elif platform.system() == 'Linux':
        for dist in get_installed_distributions():
            call("sudo -H pip install --upgrade " +
                 dist.project_name, shell=True)


def main():
    # platform_info()
    pip_upgrade_package()

main()
