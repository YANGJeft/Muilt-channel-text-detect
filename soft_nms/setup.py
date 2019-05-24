#from distutils.core import setup 
#from Cython.Build import cythonize
 #cythonize：编译源代码为C或C++，返回一个distutils Extension对象列表
#setup(ext_modules=cythonize('cpu_nms.pyx'))
#!/usr/bin/python
#python version: 2.7.3
#Filename: SetupTestOMP.py
 
# Run as:  
#    python setup.py build_ext --inplace  
   
import sys  
sys.path.insert(0, "..")  
   
from distutils.core import setup  
from distutils.extension import Extension  
from Cython.Build import cythonize  
from Cython.Distutils import build_ext
   
# ext_module = cythonize("TestOMP.pyx")  
ext_module = Extension(
                        "cpu_nms",
            ["cpu_nms.pyx"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
            )
   
setup(
    cmdclass = {'build_ext': build_ext},
        ext_modules = [ext_module], 
)




